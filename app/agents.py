import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.knowledge import AGENT_1_KNOWLEDGE, AGENT_2_KNOWLEDGE

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

def create_in_memory_rag_chain(knowledge: list[str], prompt_template: str):
    """
    Creates a RAG chain with a FAISS vector store that exists only in memory.
    This is reliable for cloud deployments with ephemeral or read-only file systems.
    """
    print("Creating in-memory vector store...")
    
    # 1. Create documents from the knowledge base
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.create_documents(knowledge)
    
    # 2. Create the FAISS vector store directly from the documents in memory
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # 3. Create the retriever and the rest of the chain
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    print("In-memory vector store and RAG chain created successfully.")
    return create_retrieval_chain(retriever, document_chain)

AGENT_1_PROMPT = """
Answer the following question based only on the provided context about general bug bounty concepts.
If the context doesn't contain the answer, state that you don't have enough information.
Keep the answer concise and helpful.

<context>
{context}
</context>

Question: {input}
"""

AGENT_2_PROMPT = """
You are a helpful assistant for our bug bounty platform. Answer the user's question about how to use the website based on the provided context.

<context>
{context}
</context>

Question: {input}
"""

print("Initializing Agent 1 Chain...")
agent_1_chain = create_in_memory_rag_chain(AGENT_1_KNOWLEDGE, AGENT_1_PROMPT)

print("Initializing Agent 2 Chain...")
agent_2_chain = create_in_memory_rag_chain(AGENT_2_KNOWLEDGE, AGENT_2_PROMPT)

print("All agents initialized.")