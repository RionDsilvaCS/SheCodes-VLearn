from typing import Union
from pydantic import BaseModel
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


app = FastAPI()

llm = ChatOllama(model="llama3")

bio_db_pth = r"C:\Users\dsilv\OneDrive\Desktop\proj\RL\RevaHack\Backend\vector_DB\biology_db"
science_db_pth = r"C:\Users\dsilv\OneDrive\Desktop\proj\RL\RevaHack\Backend\vector_DB\science_db"

embeddings = HuggingFaceEmbeddings()

bio_db = FAISS.load_local(bio_db_pth, embeddings, allow_dangerous_deserialization=True)
bio_retriever = bio_db.as_retriever(search_kwargs={"k": 5})

bio_template = """
            As a professor in Biology subject, you need to answer the questions with most precise description in short
            The question is given below

            Whenever you are responding to the student, address them by the name given below.
            name: Rion
            
            Question: {question}

            Answer the question based only on the following context:
            {context}

            """

bio_prompt = ChatPromptTemplate.from_template(bio_template)

bio_chain = (
        {"context": bio_retriever, "question": RunnablePassthrough()}
        | bio_prompt
        | llm
        | StrOutputParser()
    )


sci_db = FAISS.load_local(science_db_pth, embeddings, allow_dangerous_deserialization=True)
sci_retriever = sci_db.as_retriever(search_kwargs={"k": 5})

sci_template = """
            As a professor in Science subject, you need to answer the questions with most precise description in short
            The question is given below

            Whenever you are responding to the student, address them by the name given below.
            name: Rion
            
            Question: {question}

            Answer the question based only on the following context:
            {context}

            """

sci_prompt = ChatPromptTemplate.from_template(sci_template)

sci_chain = (
        {"context": sci_retriever, "question": RunnablePassthrough()}
        | sci_prompt
        | llm
        | StrOutputParser()
    )

class Chatllama3(BaseModel):
    prompt: str
    name: str


@app.get("/")
def read_root():
    return {"Hello": "Class"}

@app.post("/chat_science/")
def can_chat_science(chat:Chatllama3):
    return sci_chain.invoke(chat.prompt)

@app.post("/chat_biology/")
def can_chat_biology(chat:Chatllama3):                                                
    return bio_chain.invoke(chat.prompt)