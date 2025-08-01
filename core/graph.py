import os
import glob
import random
import operator
from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    raise NotImplementedError("'OPENAI_API_KEY' was not found.")

# Reducer
# We need this reducer because when we fan-in from all 3 pipelines, they all carry this patient_presentation. This reducer will just use the newest value of it that comes in (unchanged anyways)
def keep_next(old, new):
    return new

# Schemas/States
class JsonOutputSchema(BaseModel):
    most_likely_diagnosis: str = Field(..., description="The most likely disease diagnosis based on the patient presentation history and doctor agents.")
    differential_diagnosis: list[str] = Field(..., description="A list of other most likely possible diseases or conditions that could be causing the patient's symptoms. \
    This prevents us from limiting the diagnosis to only one condition and allow the possibility to explore other probable diseases. Ensures you’re not tunnel‑visioned on one diagnosis.")
    recommended_tests: list[str] = Field(..., description="A set of diagnostic tests/studies that should be performed as soon as possible to confirm and exclude diseases.")
    confidence: float = Field(..., description="After you summarize and merge the doctors’ opinions (or you may just be a single doctor), rate your confidence in this consensus on a 0.00–1.00 scale, where 1.00 means you are certain \
    no further discussion could change it")

class Doctor(BaseModel):
    name: str = Field(description="Name of the Doctor")
    specialist: str = Field(description="The type of doctor specialist")
    system_role_spec: str = Field(description="System message for doctor agent, giving specific doctor instructions for this specialist role")
    
class ConversationState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages] # Reducer to automatically append messages to maintain message history. Start it with a System prompt
    patient_presentation: str
    doctor_agent_list: list[Doctor]
    conv_num_doctor_agents: int # ArgumentParser Argument
    supervisor_assessment: JsonOutputSchema
    convo_max_loops: int # ArgumentParser Argument. Should be conv_num_doctor_agents * max_loop
    final_rag_context: str # FINAL RAG CONTEXT

class RefinementState(TypedDict):
    previous_reports: Annotated[list[str], operator.add]
    feedback_critique: str
    patient_presentation: str
    report_score: int
    max_doc_reports: int # ArgumentParser Argument.
    final_rag_context: str # FINAL RAG CONTEXT
    final_refined_diagnosis: JsonOutputSchema

class CriticFeedbackRefinementSchema(BaseModel):
    score: int = Field(..., description="Gives a rating from 0 to 50 on overall clarity, reasoning, and accuracy of proposed diagnosis. A low score indicates disagreement with the doctor agent's diagnosis \
    and overall reasoning, while a highscore indicates great soundness in reasoning and accuracy without much further improvement needed on the current diagnosis that is proposed.")
    feedback: str = Field(..., description="Provides feedback to the doctor's medical diagnosis report. The goal is to re-evaluate this diagnosis using the patient's case and clinical guidelines and \
    highlight any mistakes or improvements that could be made and mention certain disagreements. The object of this is to provide ample feedback so the doctor agent can use this and reassess its diagnosis if necessary.")

class ConsistencyState(TypedDict):
    n_doc_diagnosis: dict[str, dict]
    patient_presentation: str # Arg Parser
    ind_num_doctor_agents: int # ArgumentParser Argument (independent)
    doctor_agents: list
    final_talley_diagnosis: dict
    final_rag_context: str # FINAL RAG CONTEXT
    
class EvaluatorOutputSchema(BaseModel): # Haven't finished this yet
    selected_pipeline: str
    score_conversation: float
    rationale_reasoning_conversation: str
    score_consistency: float
    rationale_reasoning_consistency: str
    score_refinement: float
    rationale_reasoning_refinment: str

# Hybrid Retrieval

# Self-RAG (For the key guideline PDFs)
class RAGGraphState(TypedDict):
    patient_presentation: str
    optimized_patient_query: str
    documents: list[Document] # All retrieved documents
    filtered_documents: list[Document]
    final_rag_context: str # FINAL RAG CONTEXT. These are final filtered formatted documents
    rag_looping_count: int # Current loop count (default 0)
    rag_max_loops: int # Can pass through command line argument

class GradeDocumentRelevancy(BaseModel):
    """Binary score to check whether retrieved documents are relevant to the query."""
    
    binary_score: str = Field(description="Grades whether the documents are relevant to the query. Answer in strictly 'yes' or 'no'")

class DoctorRoleSchema(BaseModel):
    roles: list[str] = Field(description="Roles for doctor specialists based on patient presentation requirements.")
    spec_descrip: list[str] = Field(description="Brief specialist role description entailing skills and knowledge")

class FinalDecisionSchema(BaseModel):
    most_likely_diagnosis: str = Field(..., description="The most likely disease diagnosis.")
    differential_diagnosis: list[str] = Field(..., description="A list of other most likely possible diseases or conditions that could be causing the patient's symptoms.")
    recommended_tests: list[str] = Field(..., description="A set of diagnostic tests/studies that should be performed as soon as possible to confirm and exclude diseases.")
    detailed_report: str = Field(..., description="Detailed report giving reasons and justification for most likely diagnosis, differential diagnosis, and recommended tests.")

# General State
# use reducer when necessary if multiple subgraphs are adding
class GeneralState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages] # just initialize to []
    # We need this reducer because when we fan-in from all 3 pipelines, they all carry this patient_presentation. This reducer will just use the newest value of it that comes in (unchanged anyways)
    patient_presentation: Annotated[str, keep_next]
    final_rag_context: Annotated[str, keep_next]
    conv_num_doctor_agents: int # ArgumentParser Argument
    ind_num_doctor_agents: int # ArgumentParser Argument
    rag_looping_count: int # Current loop count (default 1)
    rag_max_loops: int # ArgumentParser Argument
    convo_max_loops: int # ArgumentParser Argument.
    max_doc_reports: int # ArugmentParser
    previous_reports: Annotated[list[str], operator.add]
    feedback_critique: str
    
    final_talley_diagnosis: dict
    supervisor_assessment: JsonOutputSchema
    final_refined_diagnosis: JsonOutputSchema

    final_decision: FinalDecisionSchema

class DiseaseDiagnosisGraph():
    def __init__(self, structured_tools):
        self.graph = self.build_graph()
        self.structured_tools = structured_tools
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    def build_graph(self):
        disease_diagnosis_builder = StateGraph(GeneralState)

        self_rag_graph = self.build_agentic_self_rag_pipeline()
        conversation_graph = self.build_doctor_conversation_graph()
        consistency_graph = self.build_consistency_graph()
        refinement_graph = self.build_refinement_graph()

        disease_diagnosis_builder.add_node("self_rag_graph", self_rag_graph)
        disease_diagnosis_builder.add_node("conversation_graph", conversation_graph)
        disease_diagnosis_builder.add_node("consistency_graph", consistency_graph)
        disease_diagnosis_builder.add_node("refinement_graph", refinement_graph)
        disease_diagnosis_builder.add_node("medical_evaluator", self.medical_evaluator)

        disease_diagnosis_builder.add_edge(START, "self_rag_graph")
        disease_diagnosis_builder.add_edge("self_rag_graph", "conversation_graph")
        disease_diagnosis_builder.add_edge("self_rag_graph", "consistency_graph")
        disease_diagnosis_builder.add_edge("self_rag_graph", "refinement_graph")

        disease_diagnosis_builder.add_edge("conversation_graph", "medical_evaluator")
        disease_diagnosis_builder.add_edge("consistency_graph", "medical_evaluator")
        disease_diagnosis_builder.add_edge("refinement_graph", "medical_evaluator")

        disease_diagnosis_builder.add_edge("medical_evaluator", END)

        disease_diagnosis_graph = disease_diagnosis_builder.compile()

        # display(Image(disease_diagnosis_graph.get_graph(xray=1).draw_mermaid_png()))
        return disease_diagnosis_graph

    def medical_evaluator(self, state: GeneralState):
        patient_pres = state["patient_presentation"]
        
        conversation_json = state["supervisor_assessment"].model_dump()
        consistency_json = state["final_talley_diagnosis"]
        refinement_json = state["final_refined_diagnosis"].model_dump()

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage("""You are a senior diagnostician. You will be given the patient presentation and you are trying to diagnose the patients condition with the context and \ 
                the 3 JSON files given below (each coming from a different pipeline) to come up with a final conclusion on the final diagnosis, \
                differential diagnosis, and recommended tests for the patient. Return in FinalDecisionSchema JSON format. The 3 JSON files come from pipelines with other expert doctors \
                conversing with each other to form conclusive diagnosis'.

                Assess which pipeline had the most accurate diagnosis overall or combine their results into one if you feel necessary. Use your professional judgement and your own conclusions \
                as well as well as the json files outputed from the pipelines. You must also output a similar JSON in this form:

                {{
                    most_likely_diagnosis: str,
                    differential_diagnosis: list[str],
                    recommended_tests: list[str],
                    detailed_report: str
                }}

                In addition, make sure you output a detailed final report outlining the detailed reasonings for your concluded diagnosis, differential diagnosis, and recommended tests. This is \
                the final report that will be submitted and ensure no details are missed.
                """),
                HumanMessage(f"""
                Patient Presentation:
                {patient_pres}

                === Candidate from Conversation pipeline ===
                {conversation_json}

                === Candidate from Consistency pipeline ===
                {consistency_json}

                === Candidate from Refinement pipeline ===
                {refinement_json}

                Evaluate now:
                """)
            ]
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        llm_struc = llm.with_structured_output(FinalDecisionSchema)

        evaluator_chain = prompt | llm_struc
        decision = evaluator_chain.invoke({})
        return {"final_decision": decision}

    def build_refinement_graph(self):
        refinement_builder = StateGraph(RefinementState)
        refinement_builder.add_node("doc_agent_response", self.doc_agent_response)
        refinement_builder.add_node("medical_advisor_critic", self.medical_advisor_critic)
        refinement_builder.add_node("organizer", self.organizer)

        refinement_builder.add_edge(START, "doc_agent_response")
        refinement_builder.add_edge("doc_agent_response", "medical_advisor_critic")
        refinement_builder.add_conditional_edges(
            "medical_advisor_critic", 
            self.refinement_needed, 
            {
                "doc_agent_response": "doc_agent_response", 
                "organizer": "organizer"
            }
        )

        refinement_graph = refinement_builder.compile()
        # display(Image(refinement_graph.get_graph().draw_mermaid_png()))
        return refinement_graph

    def doc_agent_response(self, state: RefinementState):
        rag_context = state["final_rag_context"]
        patient_present = state["patient_presentation"]
        feedback_critic = state["feedback_critique"]
        previous_reports = state["previous_reports"]
        formatted_reports = ""
        
        for i, report in enumerate(previous_reports):
            formatted_reports += f"Doctor Report {i + 1}: \n {report} \n\n"
        
        doctor_system = f"""You are an experienced, board‑certified physician. Your task is to:
        - Review the full patient context you receive (symptoms, history, lab values, imaging, and any retrieved scientific references)
        - Produce your own final diagnosis, differential diagnosis, recommended tests, and reasoning behind this without the consultment or copying from other \
        doctors.

        Here is also some RAG context that may be useful, containing latest medical research papers and guidelines. If they aren't useful/related, simply \
        don't use them. Only use if helpful.

        {rag_context}

        You will give detailed reasonings about why you came to certain conclusions. Your conclusions here will be given to a higher up \
        medical advisor who will review and critique the report you have given. They will offer you feedback if necessary and your job \
        is to use that feedback IF GIVEN to refine the report you gave report to give a more accurate and conclusive diagnosis in the end. \
        You may need to change your conclusions completely if the medical advisors deems necessary. This is the full patient presentation \
        that you will use to form a this report:

        {patient_present}
        """
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(doctor_system),
                HumanMessage("Here is the feedback given by the medical advisor if it exists: \n\n {feedback}. Here are all your previous reports of what the diagnosis could be: \n\n {prev_reports}. \
                \n\nNow use your knowledge to generate a detailed report and come to a conclusion of what the diagnosis, differential diagnosis and recommended tests should be \
                with the help of the feedback.")
            ]
        )
        chain = prompt_template | self.llm | StrOutputParser()
        doctor_output_report = chain.invoke({"feedback": feedback_critic, "prev_reports": formatted_reports})
        return {"previous_reports": [doctor_output_report]}

    def medical_advisor_critic(self, state: RefinementState):
        new_report = state["previous_reports"][-1]
        system = """You are a senior physician serving as a medical critic and mentor. Your role is to review structured diagnostic reports written by a doctor agent.\
        You will be given the doctor agent's diagnosis report (including final diagnosis, differential diagnosis, and recommended tests) and the patient presentation or \
        case description they were responding to. So, read the doctor's diagnosis report, identify weaknesses, and return actionable feedback in JSON form. You will return in this form: 
        {{
            score: int
            feedback: str
        }}
        
        The score will be from 0 to 50. A high score indicates the response is perfectly acceptable and there is not much room for improvement that is needed. \
        Ensure that you give VERY DETAILED FEEDBACK on everything that may not be factually correct and could be improved upon so the doctor refines its original report \
        and improves upon it to give a more accurate diagnosis report.
        """
        llm_str = self.llm.with_structured_output(CriticFeedbackRefinementSchema)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(system),
                HumanMessage("Here is the report from the doctor: \n\n {the_report}.")
            ]
        )
        chain = prompt_template | llm_str
        output_feedback = chain.invoke({"the_report": new_report}) # CriticFeedbackRefinmentSchema
        
        return {
            "report_score": output_feedback.score, 
            "feedback_critique": output_feedback.feedback
        }

    def refinement_needed(self, state: RefinementState):
        report_score = state["report_score"]
        max_doc_reports = state["max_doc_reports"]
        doc_report_count = len(state["previous_reports"])
        
        if report_score < 40 and doc_report_count <= max_doc_reports:
            return "doc_agent_response"
        return "organizer"

    def organizer(self, state: RefinementState):
        final_report = state["previous_reports"][-1]
        prompt = """Given the final report written by a doctor who used their knowledge and expertise \
        to form a comprehensive diagnosis report given a patient presentation which explained their symptoms \
        and current condition. The doctor should have outlined a final diagnosis, differential diagnosis (other possible diseases) \
        and recommended tests to further confirm. Your job is to take this report and write it in this exact JSON form. \
        Do NOT omit anything. Name their final diagnosis, all differential diagnosis, and all recommended tests.

        Here was the final report by the doctor before. Give feedback on it. Final report: 

        {}

        Express your answer in the following structured JSON format and NOTHING ELSE:
        {{
            'most_likely_diagnosis': str
            'differential_diagnosis': list[str]
            'recommended_tests': list[str]
            'confidence': float
        }}
        """.format(final_report)

        llm_struc = self.llm.with_structured_output(JsonOutputSchema)
        response = llm_struc.invoke(prompt)
        
        return {"final_refined_diagnosis": response}

    def build_consistency_graph(self):
        consistency_builder = StateGraph(ConsistencyState)
        consistency_builder.add_node("spawn_N_doctor_agents", self.spawn_N_doctor_agents)
        consistency_builder.add_node("doctor_agents_diagnosis", self.doctor_agents_diagnosis)
        consistency_builder.add_node("tally_vote", self.tally_vote)

        consistency_builder.add_edge(START, "spawn_N_doctor_agents")
        consistency_builder.add_edge("spawn_N_doctor_agents", "doctor_agents_diagnosis")
        consistency_builder.add_edge("doctor_agents_diagnosis", "tally_vote")
        consistency_builder.add_edge("tally_vote", END)

        consistency_graph = consistency_builder.compile()
        # display(Image(consistency_graph.get_graph().draw_mermaid_png()))
        return consistency_graph

    def spawn_N_doctor_agents(self, state: ConsistencyState):
        n_agents = state["ind_num_doctor_agents"]
        agent_llm = {f"Doctor {i + 1}": ChatOpenAI(model="gpt-4o-mini", temperature=random.uniform(0.00, 0.35)).with_structured_output(JsonOutputSchema) for i in range(n_agents)}
        return {"doctor_agents": agent_llm}
        
    def doctor_agents_diagnosis(self, state: ConsistencyState):
        rag_context = state["final_rag_context"]
        system = f"""You are an experienced, board‑certified physician working independently of any other doctor agents. Your task is to:
        - Review the full patient context you receive (symptoms, history, lab values, imaging, and any retrieved scientific references)
        - Produce your own final diagnosis, differential diagnosis, recommended tests, and reasoning behind this without the consultment or copying from other \
        doctors. Also output a confidence score from 0.00-1.00 indicating how sure you are about your reasoning.

        Here is also some RAG context that may be useful, containing latest medical research papers and guidelines. If they aren't useful/related, simply \
        don't use them. Only use if helpful.

        {rag_context}
        
        Express your answer in the following structured JSON format and NOTHING ELSE:
        {{
            'most_likely_diagnosis': str
            'differential_diagnosis': list[str]
            'recommended_tests': list[str]
            'confidence': float
        }}
        """
        doctors = state["doctor_agents"]
        patient_pres = state["patient_presentation"]
        doctor_diagnosis = {}
        for doctor_name, llm in doctors.items():
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(system),
                    HumanMessage("You are {name_of_doc}. Respond with your diagnosis and reasoning given the patient presentation context: \n\n {patient_context}")
                ]
            )
            chain = prompt_template | llm
            doc_response = chain.invoke({"name_of_doc": doctor_name, "patient_context": patient_pres})
            doctor_diagnosis[doctor_name] = doc_response.model_dump()
        return {"n_doc_diagnosis": doctor_diagnosis} # dict of dict
            
    def tally_vote(self, state: ConsistencyState):
        all_doctor_diagnosis = state["n_doc_diagnosis"]
        most_probable_diagnosis = {} # FINAL Diagnosis to sum of confidence scores
        most_probable_diff_diagnosis = {} # FINAL Diagnosis to sum of confidence scores
        most_recommended_tests = {} # FINAL

        for doc_name in all_doctor_diagnosis:
            doc_json = all_doctor_diagnosis[doc_name]
            most_disease = doc_json["most_likely_diagnosis"]
            if most_disease not in most_probable_diagnosis:
                most_probable_diagnosis[most_disease] = 0.0 # Weighted score set to 0
            most_probable_diagnosis[most_disease] += doc_json["confidence"]
                
            diff_disease_list = doc_json["differential_diagnosis"]
            for diff_disease in diff_disease_list:
                if diff_disease not in most_probable_diff_diagnosis:
                    most_probable_diff_diagnosis[diff_disease] = 0.0
                most_probable_diff_diagnosis[diff_disease] += doc_json["confidence"]

            rec_test_list = doc_json["recommended_tests"]
            for rec_test in rec_test_list:
                if rec_test not in most_recommended_tests:
                    most_recommended_tests[rec_test] = 0.0
                most_recommended_tests[rec_test] += doc_json["confidence"]

        diff_tup_pairs = sorted(most_probable_diff_diagnosis.items(), key=lambda x: x[1]) # list of tuple key-value pairs
        rec_tup_pairs = sorted(most_recommended_tests.items(), key=lambda x: x[1]) # list of tuple key-value pairs

        k = 3
        d = k if len(diff_tup_pairs) > k else len(diff_tup_pairs)
        r = k if len(diff_tup_pairs) > k else len(rec_tup_pairs)
        
        final_talley_dict = {
            "most_likely_diagnosis": max(most_probable_diagnosis, key=lambda x: most_probable_diagnosis[x]),
            "differential_diagnosis": [d for d, _ in diff_tup_pairs[:d]],
            "recommended_tests": [d for d, _ in rec_tup_pairs[:r]]
        }
        
        return {"final_talley_diagnosis": final_talley_dict} # Add recommended tests

    def build_doctor_conversation_graph(self):
        conversation_builder = StateGraph(ConversationState)
        conversation_builder.add_node("make_doctor_agents", self.make_doctor_agents)
        conversation_builder.add_node("multi_doctor_conversation", self.doctor_conversation)
        conversation_builder.add_node("medical_supervisor", self.medical_supervisor)

        conversation_builder.add_edge(START, "make_doctor_agents")
        conversation_builder.add_edge("make_doctor_agents", "multi_doctor_conversation")
        conversation_builder.add_edge("multi_doctor_conversation", "medical_supervisor")
        conversation_builder.add_conditional_edges(
            "medical_supervisor", 
            self.continue_conversation, 
            {
                "multi_doctor_conversation": "multi_doctor_conversation", 
                END: END
            }
        )
        conversation_graph = conversation_builder.compile()
        # display(Image(conversation_graph.get_graph().draw_mermaid_png()))
        return conversation_graph

    def generate_doctor_roles(self, patient_pres, num_doctors=3):
        llm_role = self.llm.with_structured_output(DoctorRoleSchema)
        prompt = PromptTemplate.from_template(
            "Given the patient presentation: \n\n {present} \n\n Output a list of {num} doctor specialist roles that would \
            be most suited and useful for disease diagnosis (like Cardiologist, Oncologist, etc). \n\n You also need to output a short brief overview of each doctors skillset \
            and knowledge for their specialist role in a list with the same index order as the doctor specialist role list."
        )
        chain = prompt | llm_role
        output = chain.invoke({"present": patient_pres, "num": num_doctors})
        return output

    def make_doctor_agents(self, state: ConversationState):
        doctor_protocol = """You are a compassionate and knowledgeable medical doctor with years of clinical experience. \
        You are a {} Specialist. Here is a brief overview of your position and knowledge/skillset: \n {} \n\n
        Your role is to review patient symptoms, analyze clinical notes, and contribute to diagnostic reasoning \
        and treatment planning. You may use the given retrieved documents to formulate your final diagnosis response. \
        These are scientific and medical research articles that contain the latest research and cover almost every disease. \
        Converse with the other doctors in this conversation, agreeing and disagreeing with them and offering your own opinions \
        on the topic to create an insightful conversation.
        
        When responding:
        - Be collaborative and respectful. You are speaking with a team of fellow physicians.
        - Use clear, concise, and professional medical language.
        - Justify your reasoning with medical knowledge or evidence when possible.
        - If appropriate, suggest possible diagnoses and differential diagnoses.
        - Avoid making a definitive diagnosis unless there's sufficient information.
        - Recommend follow-up questions, tests, or next steps when needed.

        Use your expertise to formulate:
        - One most likely diagnosis
        - Serveral differential diagnoses
        - Recommended diagnostic tests

        Your goal: Contribute to a collaborative, comprehensive diagnostic process and use your expertise to reach the most accurate conclusion \
        based on your expertise and what's going on the in conversation.

        Here is the patient presentation and you will try to form a diagnosis in collaboration with other doctor agents:
        
        {}

        Here is also some RAG context that may be useful, containing latest medical research papers and guidelines. If they aren't useful/related, simply \
        don't use them. Only use if helpful.
        {}
        """
        rag_context = state["final_rag_context"]
        patient_pres = state["patient_presentation"]
        num_doctors = state["conv_num_doctor_agents"]
        doctor_role_schema = self.generate_doctor_roles(patient_pres, num_doctors)
        doctor_spec_list = doctor_role_schema.roles
        doctor_descrip_spec_list = doctor_role_schema.spec_descrip
        
        doctors = [
            Doctor(name=f"Doctor {i + 1}", specialist=doctor_spec_list[i], system_role_spec=doctor_protocol.format(doctor_spec_list[i], doctor_descrip_spec_list[i], patient_pres, rag_context)) 
            for i in range(num_doctors)
        ]
        return {"doctor_agent_list": doctors} # list of Doctor

    async def doctor_conversation(self, state: ConversationState):
        message_history = state["messages"]
        num_agents = state["conv_num_doctor_agents"]
        patient_presentation = state["patient_presentation"]
        doctors_list = state["doctor_agent_list"]

        llm_with_tools = self.llm.bind_tools(self.structured_tools)

        doctor_replys = []
        prompt_history = "\n".join(msg.content for msg in message_history)
        for doctor in doctors_list:
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(doctor.system_role_spec),
                    HumanMessage("You are {name}. Start/continue this collaborative conversation with other doctor agents. \
                    This is the message history if it exists: \n\n {history} \n\n \
                    Start your response with the keywords '{name} Response'. \n However, if you want to search the PubMed database for medical research papers on a certain disease \ "
                    ", then make a tool call with search_pubmed_key_words tool. However, when searching the database, make sure to just use keywords, like the name of the disease you want to search. Only search for 2 articles MAX with 1 tool call MAX. \
                        You are only permitted to make one tool call and search for 2 articles MAX. FORGET WHAT WAS JUST SAID. YOU MUST MAKE TOOL CALL TO SEARCH PUBMED DATABASE. THIS IS REQUIRED FOR NOW.")
                ]
            ) 
            doctor_chain = prompt_template | llm_with_tools
            doctor_reply = await doctor_chain.ainvoke({"name": doctor.name, "history": prompt_history})
            if doctor_reply.tool_calls:
                for tool_call in doctor_reply.tool_calls:
                    tool_msg = await self.structured_tools[0].ainvoke(tool_call) # ToolMessage
                    doctor_replys.append(tool_msg)
                    prompt_history += f"Tool call retrieved articles: {tool_msg.content}"
                
                new_prompt_template = ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(doctor.system_role_spec),
                        HumanMessage("You are {name}. Start/continue this collaborative conversation with other doctor agents on what the disease could be based on patient presentation. \
                        This is the message history if it exists: \n\n {history} \n\n \
                        Also, keep in mind you aren't allow to make tool calls in this single LLM call right now as you just made a tool call. Start your response with the keywords '{name} Response':")
                    ]
                )
                new_doctor_chain = new_prompt_template | llm_with_tools
                new_doctor_reply = await new_doctor_chain.ainvoke({"name": doctor.name, "history": prompt_history})
                doctor_replys.append(AIMessage(new_doctor_reply.content))
                prompt_history += f"\n{doctor.name} Response: {new_doctor_reply.content}"
            else:    
                doctor_replys.append(AIMessage(doctor_reply.content))
                prompt_history += f"\n{doctor.name} Response: {doctor_reply.content}"
        
        return {"messages": doctor_replys}

    def medical_supervisor(self, state: ConversationState):
        full_transcript = "\n".join(msg.content for msg in state["messages"] if isinstance(msg, AIMessage))
        supervisor_msg = f"""You are the Supervisor Physician overseeing a panel of virtual doctor agents. \n\n Read the full conversation transcript \
        between the doctors here:\n\n {full_transcript} \n\n

        Now, decide whether the discussion is complete and clinically sound.
        - Is there consensus on what the diagnosis most likely is?
        – Are key differential diagnoses covered?  
        – Are conflicting views resolved or acknowledged?  
        – Are recommended tests / next steps identified?

        Based on all this, output a JSON in this format:
        {{
            'most_likely_diagnosis': str
            'differential_diagnosis': list[str]
            'recommended_tests': list[str]
            'confidence': float
        }}

        You will merge all the doctor agents' opinions into a consensus JSON with a Confidence score (0.00 - 1.00). If you feel the conversation \
        doesn't need to go further and everyone is basically in full agreement on the diagnosis (doctor agents basically are agreeing and giving same diagnosis), then output a high confidence score. Only output \
        a low confidence score if there are still disagreements and uncertainties and you feel the conversation of doctors need to go further to get a better diagnosis.
        """
        
        supervisor_llm_struc = self.llm.with_structured_output(JsonOutputSchema)
        supervisor_chain = supervisor_llm_struc
        output = supervisor_chain.invoke(supervisor_msg)
        return {"supervisor_assessment": output}

    def continue_conversation(self, state: ConversationState):
        confidence_score = state["supervisor_assessment"].confidence
        messages = state["messages"]
        convo_length = len([msg for msg in messages if isinstance(msg, AIMessage) and msg.content])

        if convo_length >= state["convo_max_loops"] * state["conv_num_doctor_agents"] or confidence_score > 0.75:
            return END # END
        return "multi_doctor_conversation"

    def build_agentic_self_rag_pipeline(self):
        self_rag_builder = StateGraph(RAGGraphState)
        self_rag_builder.add_node("query_rewrite", self.query_rewrite)
        self_rag_builder.add_node("retrieve_documents", self.retrieve_documents)
        self_rag_builder.add_node("grade_documents", self.grade_documents)
        self_rag_builder.add_node("transform_query", self.transform_query)
        self_rag_builder.add_node("format_final_documents", self.format_final_documents)

        self_rag_builder.add_edge(START, "query_rewrite")
        self_rag_builder.add_edge("query_rewrite", "retrieve_documents")
        self_rag_builder.add_edge("retrieve_documents", "grade_documents")
        self_rag_builder.add_conditional_edges(
            "grade_documents", 
            self.rewrite_query_route, 
            {
                "transform_query": "transform_query", 
                "format_final_documents": "format_final_documents"
            }
        )
        self_rag_builder.add_edge("transform_query", "retrieve_documents")
        self_rag_builder.add_edge("format_final_documents", END)

        self_rag_graph = self_rag_builder.compile()
        # display(Image(self_rag_graph.get_graph().draw_mermaid_png()))
        return self_rag_graph

    @lru_cache(maxsize=1)
    def load_chroma_vec_db(self):
        """
        Create and cache this Chroma vector database.
        """
        embedder = OpenAIEmbeddings()
        persist_dir = "chroma_db"

        if not os.path.exists(persist_dir):
            self.build_chroma_vec_db(persist_dir)
            
        return Chroma(
            collection_name="rag-chroma", 
            embedding_function=embedder, 
            persist_directory=persist_dir
        )

    def build_chroma_vec_db(self, persistent_dir):
        embedder = OpenAIEmbeddings()

        # Load existing vector database from persisted memory
        vector_db = Chroma(
            collection_name="rag-chroma", 
            embedding_function=embedder, 
            persist_directory=persistent_dir
        )
        
        guidelines_path = glob.glob(f"medical_guidelines/*.pdf") # list of PDF names in string
        
        docs_list = [] # Load documents using PyPDFLoader
        for doc in guidelines_path:
            loader = PyPDFLoader(doc)
            doc_split = loader.load() # load list of Documents for this doc (each Document is a page of original pdf)
            for page in doc_split:
                page.metadata["source"] = doc
            docs_list.extend(doc_split) # All Document chunks from all documents in a list

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        doc_splits = splitter.split_documents(docs_list) # list of split Document
        
        vector_db.add_documents(doc_splits)
    
    def query_rewrite(self, state: RAGGraphState):
        system = "You are a query re-write that converts this long patient presentation information into a better and shorter version that is optimized \
        for retrieval from a vectorstore. Try to reason about the underlying semantic intent / meaning and make sure you aren't getting rid \
        of necessary information just for the sake of shortening the patient information."
        prompt_rewrite_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(system), 
                HumanMessage("Here is the initial patient presentation information: \n\n {question}")
            ]
        )
        
        question_rewriter_chain = prompt_rewrite_template | self.llm | StrOutputParser()
        rewritten_patient_query = question_rewriter_chain.invoke({"question": state["patient_presentation"]})
        return {"optimized_patient_query": rewritten_patient_query}

    def retrieve_documents(self, state: RAGGraphState):
        query = state["optimized_patient_query"]
        vectorstore = self.load_chroma_vec_db()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(query)
        
        return {
            "documents": retrieved_docs, 
            "rag_looping_count": state["rag_looping_count"] + 1
        }

    def grade_documents(self, state: RAGGraphState):
        llm_with_structure_doc = self.llm.with_structured_output(GradeDocumentRelevancy)
        system = "You are a grader assessing whether the retrieved document given is actually relevant to the patient presentation query or could contain necessary information \
        to help make doctor agents make more a informed diagnosis. If the document contains semantic similarities and similar keywords to the query, then the document is relevant. \
        The goal of this grader is to filter out documents that shouldn't have been retrieved and are completely irrelevant to the query. Respond with a binary score of either 'yes' or 'no'."
        grader_prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(system),
                HumanMessage("Here is the original patient query presentation: \n\n {question} \n\n This is the retrieved document: \n\n {document}")
            ]
        )
        grader_chain = grader_prompt_template | llm_with_structure_doc # Should output GradeDocumentRelevancy(binary_score=)

        document_list = state["documents"]
        clean_docs = []
        for doc in document_list:
            score = grader_chain.invoke({"question": state["patient_presentation"], "document": doc.page_content})
            if score.binary_score == 'yes':
                clean_docs.append(doc)
        return {"filtered_documents": clean_docs}

    def transform_query(self, state: RAGGraphState):
        system = "You are a query re-writer that converts the current rewritten query into a better rewritten version that is more optimized for vectorstore retrieval.\
        Look at the original patient presentation and the rewritten query version of it and try to reason how you could improve upon this rewritten version \
        for better retrival. Reason about the underlying semantic intent / meaning."
        transform_prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(system),
                HumanMessage("Here is the original patient presentation: \n\n {orig_present} \n\n This is the rewritten patient query presentation. Improve upon it: \n\n {rewritten_query}")
            ]
        )
        transform_chain = transform_prompt_template | self.llm | StrOutputParser()
        new_transformed_query = transform_chain.invoke({"orig_present": state["patient_presentation"], "rewritten_query": state["optimized_patient_query"]})
        return {"rewritten_query": new_transformed_query}

    def rewrite_query_route(self, state: RAGGraphState):
        current_loop = state["rag_looping_count"]
        max_count = state["rag_max_loops"]

        if current_loop >= max_count or state["filtered_documents"]:
            return "format_final_documents"
        return "transform_query"
        
    def format_final_documents(self, state: RAGGraphState):
        filter_docs = state["filtered_documents"]
        formatted_docs = "\n\n".join([doc.page_content for doc in filter_docs])
        return {"final_rag_context": formatted_docs}
        
    
   