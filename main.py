import asyncio
import argparse
from mcp_client import MCPPubMedClient
from typing import Any
from mcp.types import Tool as MCPTool
from langchain_core.tools.structured import StructuredTool
from graph import DiseaseDiagnosisGraph

async def main():
    async with MCPPubMedClient("http://127.0.0.1:8000/mcp") as client:
        all_tools = await client.list_mcp_tools() # list[MCPTool]
        search_keyword_tool = [tool for tool in all_tools if tool.name == "search_pubmed_key_words"][0] # MCPTool

        async def convert_mcp_tool_to_langchain_tool(tool: MCPTool):
            async def tool_call(**arguments: dict[str, Any]):
                tool_call_result = await client.session.call_tool(tool.name, arguments)
                str_tool_call_result = [article.text for article in tool_call_result.content] # This would make this specific to search_keyword_tool tool
                return str_tool_call_result
            
            return StructuredTool(
                name=tool.name,
                description=tool.description or "",
                args_schema=tool.inputSchema, # What arguments tool takes
                coroutine=tool_call # How the ToolNode/agent can actually call the function
            )
        
        pubmed_search_tool = [await convert_mcp_tool_to_langchain_tool(search_keyword_tool)] # loop convert_mcp_tool_to_langchain_tool across MCPTools to get many StructuredTools

        # result = await pubmed_search_tool[0].ainvoke({'key_words': 'CRISPR', 'num_results': 2})
        # print(result)

        async def search_pubmed(key_words: str, num_results: int):
            """
            Search for medical research articles on PubMed database using key words.
            """
            articles = await client.session.call_tool(
                "search_pubmed_key_words", 
                {
                    "key_words": key_words,
                    "num_results": num_results
                }
            ).content

            return [article.text for article in articles]
        
        cli_args = parse_arguments()

        disease_diagnosis_graph = DiseaseDiagnosisGraph(search_pubmed_func=search_pubmed, structured_tools=pubmed_search_tool)

        test = "\"{\\n  \\\"Basic Information\\\": \\\"The patient is an Omani female in her 30s from Sohar.\\\",\\n\\n  \\\"Clinical Presentation\\\": \\\"The patient presented with complaints of difficulty and painful swallowing (odynophagia) for one year. The dysphagia was of gradual onset and progressive, starting as difficulty swallowing solids, which then recently developed into difficulty swallowing both solids and liquids with feelings of food and drinks 'being stuck in her throat'. She reported having easy fatiguability as well as unintentional weight loss. She denied any fever, abdominal pain, nausea, vomiting, heartburn, hematemesis, melena, joint pain, skin rashes or Raynaud's phenomenon. The patient had a history of chronic iron-deficiency anemia of several years and had not taken iron supplements previously. The patient denied any abnormalities in the menstrual cycle and had no pregnancies in the past. The surgical history was unremarkable.\\\",\\n\\n  \\\"Physical Examination\\\": \\\"The patient appeared tired but was fully conscious and cooperative. The patient was vitally stable. Conjunctival pallor, glossitis and angular stomatitis were noted. On hand examination, the patient also had koilonychia and pallor. No lymphadenopathy or skin rashes were seen. On abdominal examination, no tenderness or organomegaly was present, but a round, firm, smooth mass (diameter of 3 cm approximately) over the lower abdomen was palpated. It was immobile and non-tender.\\\",\\n\\n  \\\"Past Medical History\\\": \\\"The patient had a history of chronic iron-deficiency anemia of several years. A family history of G6PD deficiency was reported to be present in siblings. No family history of gastrointestinal malignancy or autoimmune disorders was present.\\\",\\n\\n  \\\"Initial Test Results\\\": \\\"Laboratory studies on admission showed microcytic hypochromic anemia secondary to iron deficiency. No leukocytosis or leukopenia was noted. Thyroid, liver and renal function tests were all within normal levels. Autoimmune workup (including rheumatoid factor, anti-nuclear antibody, erythrocyte sedimentation rate and antibodies specific to Sjogren's syndrome and scleroderma) was unremarkable. Ultrasound of the neck and abdomen was unremarkable. Ultrasound of the pelvis revealed a uterine fibroid measuring 5 \\u00d7 5.5 cm.\\\"\\n}\""
        final_answer_state = await disease_diagnosis_graph.graph.ainvoke(
            {
                "messages": [],
                "patient_presentation": test,
                "conv_num_doctor_agents": cli_args.conv_num_doctor_agents, # 3
                "ind_num_doctor_agents": cli_args.ind_num_doctor_agents, # 6
                "rag_looping_count": 0, # 0
                "rag_max_loops": cli_args.rag_max_loops, # 2
                "convo_max_loops": cli_args.convo_max_loops, # 2
                "max_doc_reports": cli_args.max_doc_reports, # 4
                "feedback_critique": "",
                "previous_reports": []
            }
        )

        # TESTING
        print(final_answer_state["final_decision"])
        print("=========")
        print(final_answer_state["final_refined_diagnosis"])
        print("=========")
        print(final_answer_state["supervisor_assessment"])
        print("=========")
        print(final_answer_state["final_talley_diagnosis"])
        print("=========")
        print(final_answer_state["final_rag_context"])
        print("=========")
        print(final_answer_state["patient_presentation"])

def parse_arguments():
    parser = argparse.ArgumentParser(description="Multi-Agent Settings")
    parser.add_argument(
        "--conv_num_doctor_agents",
        type=int,
        default=3,
        help="Number of doctor agents in the conversation during disease diagnosis"
    )

    parser.add_argument(
        "--ind_num_doctor_agents",
        type=int,
        default=6,
        help="Number of doctor agents making independent diagnosis"
    )

    parser.add_argument(
        "--rag_max_loops",
        type=int,
        default=2,
        help="How many times re-retrieve documents during agentic RAG"
    )

    parser.add_argument(
        "--convo_max_loops",
        type=int,
        default=2,
        help="How many times to loop the doctor agent conversation"
    )
    
    parser.add_argument(
        "--max_doc_reports",
        type=int,
        default=4,
        help="How many times doctor agent can ask critic to refine report"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    asyncio.run(main())