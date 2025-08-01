import uvicorn
from fastapi import FastAPI
from app.dependencies import lifespan
from app.api.diagnose import router as diagnose_router

app = FastAPI(lifespan=lifespan)
app.include_router(diagnose_router)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )


# async def main():
#     async with MCPPubMedClient("http://127.0.0.1:8000/mcp") as client:
#         pubmed_search_tool = await get_structured_tools(client=client) # loop convert_mcp_tool_to_langchain_tool across MCPTools to get many StructuredTools

#         # result = await pubmed_search_tool[0].ainvoke({'key_words': 'CRISPR', 'num_results': 2})
#         # print(result)
        
#         cli_args = parse_arguments()

#         disease_diagnosis_graph = DiseaseDiagnosisGraph(structured_tools=pubmed_search_tool)

#         test = "\"{\\n  \\\"Basic Information\\\": \\\"The patient was a 2-month-old female infant born after 37 weeks of gestation with a birth weight of 2730 g.\\\",\\n  \\\"Clinical Presentation\\\": \\\"At 2 months, she experienced her first episode of syncope due to repetitive TdPs that degenerated into VF. Her ECG revealed typical T wave alternans, markedly prolonged QT-intervals (RR\\u2009=\\u2009570 ms, QT\\u2009=\\u2009501 ms, QTc\\u2009=\\u2009664 ms), 2:1 atrio-ventricular (AV) block, and recurrence of TdP. The echocardiography showed no congenital heart defects nor hypertrophy. Mexiletine was administered as an initial therapy, which resolved the 2:1 AV block to a 1:1 conduction. Propranolol was also started, which suppressed recurrence of TdP. The patient also suffered from recurrent seizures unrelated to TdPs from 4 months after birth. An electroencephalogram at the age of 7 months displayed hypsarrhythmia. She showed severe developmental disability and hypotonia, and thus she was barely able to roll over at the age of 3. The patient was also diagnosed with autism spectrum disorder at the age of 2. With the medications, the patient\\u2019s ECG at the age of 5 showed slightly prolonged QTc (RR\\u2009=\\u2009559 ms, QT\\u2009=\\u2009346 ms, QTc\\u2009=\\u2009462 ms). Since the pharmacotherapy successfully suppressed her TdP, implantable cardioverter-defibrillator (ICD) was not implanted. Unfortunately, the patient suddenly passed away at 5 years old during a nap.\\\",\\n  \\\"Physical Examination\\\": \\\"The patient\\u2019s face was characterized by dysmorphic features such as high arched palate, full cheeks, and congenital clasped thumb, but no syndactyly. She showed severe developmental disability and hypotonia, and was barely able to roll over at the age of 3.\\\",\\n  \\\"Past Medical History\\\": \\\"There were no findings that indicated hypoglycemia or immunodeficiency related to TS. Her family history was negative for SCD, LQTS, arrhythmia, or neurological abnormalities.\\\",\\n  \\\"Initial Test Results\\\": \\\"Her ECG revealed typical T wave alternans, markedly prolonged QT-intervals (RR\\u2009=\\u2009570 ms, QT\\u2009=\\u2009501 ms, QTc\\u2009=\\u2009664 ms). Echocardiography showed no congenital heart defects nor hypertrophy. An electroencephalogram at the age of 7 months displayed hypsarrhythmia. The patient\\u2019s ECG at the age of 5 showed slightly prolonged QTc (RR\\u2009=\\u2009559 ms, QT\\u2009=\\u2009346 ms, QTc\\u2009=\\u2009462 ms).\\\"\\n}\""
#         final_answer_state = await disease_diagnosis_graph.graph.ainvoke(
#             {
#                 "messages": [],
#                 "patient_presentation": test,
#                 "conv_num_doctor_agents": cli_args.conv_num_doctor_agents, # 3
#                 "ind_num_doctor_agents": cli_args.ind_num_doctor_agents, # 6
#                 "rag_looping_count": 0, # 0
#                 "rag_max_loops": cli_args.rag_max_loops, # 2
#                 "convo_max_loops": cli_args.convo_max_loops, # 2
#                 "max_doc_reports": cli_args.max_doc_reports, # 4
#                 "feedback_critique": "",
#                 "previous_reports": []
#             }
#         )

#         # TESTING
#         print(final_answer_state["final_decision"])
#         print("=========")
#         print(final_answer_state["final_refined_diagnosis"])
#         print("=========")
#         print(final_answer_state["supervisor_assessment"])
#         print("=========")
#         print(final_answer_state["final_talley_diagnosis"])
#         print("=========")
#         print(final_answer_state["final_rag_context"])
#         print("=========")
#         print(final_answer_state["messages"])

# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Multi-Agent Settings")
#     parser.add_argument(
#         "--conv_num_doctor_agents",
#         type=int,
#         default=3,
#         help="Number of doctor agents in the conversation during disease diagnosis"
#     )

#     parser.add_argument(
#         "--ind_num_doctor_agents",
#         type=int,
#         default=6,
#         help="Number of doctor agents making independent diagnosis"
#     )

#     parser.add_argument(
#         "--rag_max_loops",
#         type=int,
#         default=2,
#         help="How many times re-retrieve documents during agentic RAG"
#     )

#     parser.add_argument(
#         "--convo_max_loops",
#         type=int,
#         default=2,
#         help="How many times to loop the doctor agent conversation"
#     )
    
#     parser.add_argument(
#         "--max_doc_reports",
#         type=int,
#         default=4,
#         help="How many times doctor agent can ask critic to refine report"
#     )

#     args = parser.parse_args()
#     return args

# if __name__ == "__main__":
#     asyncio.run(main())

