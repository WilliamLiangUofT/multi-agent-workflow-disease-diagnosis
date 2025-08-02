import json
import requests
import argparse
from medicaldata import MedicalDataset

def main():
    cli_args = parse_arguments()
    dataset = MedicalDataset("rare_disease_302.json")
    url = "http://localhost:8001/diagnosis/"

    conv_doc_agents = cli_args.conv_num_doctor_agents
    ind_doc_agents = cli_args.ind_num_doctor_agents
    rag_max_loops = cli_args.rag_max_loops
    convo_max_loops = cli_args.convo_max_loops
    max_doc_reports = cli_args.max_doc_reports

    for patient_presentation, ground_truth_diagnosis in dataset[:2]:
        response = call_diagnose_single_case(
            patient_presentation,
            url,
            conv_doc_agents,
            ind_doc_agents,
            rag_max_loops,
            convo_max_loops,
            max_doc_reports
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        print(f"Ground Truth {ground_truth_diagnosis}")
        print()


def call_diagnose_single_case(
        patient_presentation: str,
        url: str,
        conv_num_doctor_agents: int,
        ind_num_doctor_agents: int,
        rag_max_loops: int,
        convo_max_loops: int,
        max_doc_reports: int
):
    
    payload = {
        "messages": [],
        "patient_presentation": patient_presentation,
        "conv_num_doctor_agents": conv_num_doctor_agents, # 3
        "ind_num_doctor_agents": ind_num_doctor_agents, # 6
        "rag_looping_count": 0, # 0
        "rag_max_loops": rag_max_loops, # 2
        "convo_max_loops": convo_max_loops, # 2
        "max_doc_reports": max_doc_reports, # 4
        "feedback_critique": "",
        "previous_reports": []
    }

    # Headers for JSON content. We will convert payload to JSON string. Headers here specify the format of the data being sent (JSON).
    headers = {
        "Content-Type": "application/json",
    }

    response = requests.post(url=url, data=json.dumps(payload), headers=headers)
    return response

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
    main()