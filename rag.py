import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings

# Load env variables
load_dotenv("/etc/secrets/secret.env")

app = FastAPI()

# Memory & Chain Setup
loader = TextLoader("mydata.txt")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, memory=memory, verbose=True,condense_question_llm=None
)

class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(req: QueryRequest):
    print(f"Received question: {req.question}")
    try:
        result = qa_chain({"question": req.question})
        print("Got result from chain.")
        return {"answer": result["answer"]}
    except Exception as e:
        print("Error:", str(e))
        return {"answer": "Something went wrong on the server."}

@app.get("/")
async def health():
    return {"status" : "ok"}


@app.post("/post")
async def post(req: QueryRequest):
    return {"status" : "Haa bhe hutyee"}
    
