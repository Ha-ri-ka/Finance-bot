from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("Ripsu.csv")
embeddings = OllamaEmbeddings(model='mxbai-embed-large')

db_loc = "./chroma_langchain_db"
add_documents = not os.path.exists(db_loc)

if add_documents:
    documents=[]
    ids=[]

    for i,row in df.iterrows():
        document = Document(
            page_content=row["category"] + " " + row["content"],
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="Ripradaman",
    persist_directory=db_loc,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents,ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={ "k" : 1} #number of relevant rows from csv to look up to send to LLM
)