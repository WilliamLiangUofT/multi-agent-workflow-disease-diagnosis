from pydantic import BaseModel

class DiagnoseIn(BaseModel):
    messages: list
    patient_presentation: str
    conv_num_doctor_agents: int
    ind_num_doctor_agents: int
    rag_looping_count: int
    rag_max_loops: int
    convo_max_loops: int
    max_doc_reports: int
    feedback_critique: str
    previous_reports: list

class DiagnoseOut(BaseModel):
    most_likely_diagnosis: str
    differential_diagnosis: list[str]
    recommended_tests: list[str]
    detailed_report: str
    
