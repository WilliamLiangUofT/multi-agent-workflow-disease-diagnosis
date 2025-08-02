from fastapi import APIRouter, Depends
from app.dependencies import get_tools
from core.graph import DiseaseDiagnosisGraph
from app.api.schemas import DiagnoseIn, DiagnoseOut

router = APIRouter()

# Define API Endpoint
@router.post("/diagnosis", response_model=DiagnoseOut)
async def diagnose_single_case(req: DiagnoseIn, tools: list = Depends(get_tools)):
    disease_diagnosis_graph = DiseaseDiagnosisGraph(structured_tools=tools)
    final_answer_state = await disease_diagnosis_graph.graph.ainvoke(
        {
            "messages": [],
            "patient_presentation": req.patient_presentation,
            "conv_num_doctor_agents": req.conv_num_doctor_agents, # 3
            "ind_num_doctor_agents": req.ind_num_doctor_agents, # 6
            "rag_looping_count": 0, # 0
            "rag_max_loops": req.rag_max_loops, # 2
            "convo_max_loops": req.convo_max_loops, # 2
            "max_doc_reports": req.max_doc_reports, # 4
            "feedback_critique": "",
            "previous_reports": []
        }
    )

    final_result_dict = final_answer_state["final_decision"].model_dump()
    return DiagnoseOut(**final_result_dict)

