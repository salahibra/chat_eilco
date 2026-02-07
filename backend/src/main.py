from Knowledge_base import Knowledge_base
from RAG import RAG
import os
from config import Config


build_knowledge_base = True
run_rag = False



config = Config()

if build_knowledge_base:
    kb = Knowledge_base(
        dir_files=config.dir_files, 
        export_type=config.export_type, 
        embedding_model_id=config.embedding_model_id, 
        top_k=config.top_k,
        persist_directory=config.persist_directory
    )
    #documents = kb.loader()
    #kb.splitter(documents)
    #kb.ingestion()
    relevant_docs = kb.retriever.invoke(config.question)



    print(f"Number of relevant docs: {len(relevant_docs)}")
    for i, doc in enumerate(relevant_docs):
        print(f"Document {i+1} content: {doc.page_content}\n")
        print("metadata:", doc.metadata)


if run_rag:
    kb = Knowledge_base(
        dir_files=config.dir_files, 
        export_type=config.export_type, 
        embedding_model_id=config.embedding_model_id, 
        top_k=config.top_k,
        persist_directory=config.persist_directory
    )
    rag = RAG(model_api_url=config.llm_api_url, model_name=config.llm_name, retriever=kb.retriever, prompt=config.prompt)
    query = "quels sont les prérequis, les objectifs et le programme pour le module Microbiologie générale?"
    response = rag.response_generator(query)
    print(f"Response from Model: \n {response['choices'][0]['message']['content']}")