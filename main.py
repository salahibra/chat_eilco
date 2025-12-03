from Knowledge_base import Knowledge_base
from RAG import RAG
files_paths = [
    "/home/salah/EILCO/PIC/chateilco/docx_formation/Guide-des-Etudes_EIL_Cycle-Preparatoire_2024-2025.docx",
    # "/home/salah/EILCO/PIC/chateilco/docx_formation/REGLEMENT INTERIEUR 2024-2025.docx",
    # "/home/salah/EILCO/PIC/chateilco/docx_formation/REGLEMENT INTERIEUR 2024-2025.docx", 
               
]
# kb = Knowledge_base(list_file_paths=files_paths)
# documents = kb.loader()
# print the first document loaded just 5 lines
# print(len(documents))
# print(documents[0].page_content[:50])
# print(documents[0].metadata)
# documents = [Document(page_content="This is a test document.", metadata={"source": "test"}), Document(page_content="This is another test document.", metadata={"source": "test2"})]
# chunks  = kb.splitter(documents)
# print(f"Number of chunks: {len(splitter)}")
# print("split 1 : ", chunks[0].page_content)
# print(type(chunks[0]))
# print("split 1 metadata : ", chunks[0].metadata)


# embeddings = kb.converter([chunks[0]])
# print(f"Number of embeddings: {len(embeddings)}")
# print("Embedding 1 : ", embeddings[0])
# vectorstore = kb.storer(chunks)


# print("Vectorstore info: ", vectorstore)
# query = "Quels sont les critères d'admission pour le cycle préparatoire ?"
# docs = vectorstore.similarity_search(query, k=3)
# print(f"Top 3 similar documents for the query '{query}':")
# for i, doc in enumerate(docs):
#     print(f"Document {i+1}:")
#     print(doc.page_content)
#     print("Metadata:", doc.metadata)
#     print()


rag = RAG()
rag.load_knowledge_base()
query = "Quels sont les critères d'admission pour le cycle préparatoire ?"
docs = rag.retriever(query)
# print(f"Retrieved Documents: {docs}")
augmented_prompt = rag.prompt_augmentation(docs, query)
# print(f"Augmented Prompt: {augmented_prompt}")
response = rag.response_generator(augmented_prompt)
print(f"Response from Model: \n {response['choices'][0]['message']['content']}")