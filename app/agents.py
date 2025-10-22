import os
from dotenv import load_dotenv
import faiss 

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser


from app.knowledge import AGENT_1_KNOWLEDGE, AGENT_2_KNOWLEDGE


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

FAISS_INDEX_PATH_AGENT_1 = "faiss_index_agent_1"
FAISS_INDEX_PATH_AGENT_2 = "faiss_index_agent_2"


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'} # Explicitly run on CPU
)

def get_or_create_rag_chain(knowledge: list[str], index_path: str, prompt_template: str):
    """
    Loads a FAISS vector store from disk if it exists, otherwise creates it
    and saves it to disk. Then creates a RAG chain.
    """
    vector_store = None
    if os.path.exists(index_path):
        print(f"Loading existing vector store from {index_path}...")
        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True # Required for FAISS loading
        )
    else:
        print(f"No vector store found. Creating a new one at {index_path}...")
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.create_documents(knowledge)
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(index_path)
        print("Vector store created and saved.")

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_template(prompt_template)

    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


def create_simple_llm_chain(prompt_template: str):
    """
    Creates a simple LLM chain without RAG/retrieval.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    def invoke_wrapper(inputs):
        result = chain.invoke(inputs)
        return {"answer": result}
    
    class SimpleChain:
        def invoke(self, inputs):
            return invoke_wrapper(inputs)
    
    return SimpleChain()

AGENT_1_PROMPT = """
Your expertise lies in Ethical Hacking with an emphasis on bug bounties. Answer questions with a reference to your expertise only.
Keep the answer helpful.
"""

AGENT_2_PROMPT = """
You are a helpful assistant for our bug bounty platform. Answer the user's question about how to use the website based on the provided context.

<context>
{context}
</context>

Question: {input}
"""


print("Initializing Agent 1 Chain...")
agent_1_chain = create_simple_llm_chain(AGENT_1_PROMPT)

print("Initializing Agent 2 Chain...")
agent_2_chain = get_or_create_rag_chain(AGENT_2_KNOWLEDGE, FAISS_INDEX_PATH_AGENT_2, AGENT_2_PROMPT)

print("All agents initialized.")