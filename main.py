from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

from graph import app_graph

app = FastAPI(
    title="Bug Bounty Chatbot",
    description="A chatbot for answering questions about bug bounties and our website.",
)

class SerializableHumanMessage(BaseModel):
    content: str
    type: str = "human"

class SerializableAIMessage(BaseModel):
    content: str
    type: str = "ai"

SerializableMessage = SerializableHumanMessage | SerializableAIMessage

def deserialize_history(messages: List[SerializableMessage]) -> List[AnyMessage]:
    """Converts serialized messages back to LangChain message objects."""
    history = []
    for msg in messages:
        if msg.type == "human":
            history.append(HumanMessage(content=msg.content))
        elif msg.type == "ai":
            history.append(AIMessage(content=msg.content))
    return history

class QueryRequest(BaseModel):
    query: str
    chat_history: List[SerializableMessage] = Field(default_factory=list)
    session_id: str = Field(..., description="Unique ID per user/session")

class QueryResponse(BaseModel):
    answer: str
    chat_history: List[SerializableMessage]


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Receives a query, chat history, and session_id.
    Uses LangGraph session-based memory for persistent conversation context.
    """
    history = deserialize_history(request.chat_history)
    inputs = {"query": request.query, "chat_history": history}

    # ✅ Pass session_id as the thread identifier for persistent memory
    result = app_graph.invoke(
        inputs,
        config={"configurable": {"thread_id": request.session_id}}
    )

    # Serialize the updated history for JSON response
    updated_history_serializable = [
        SerializableHumanMessage(content=msg.content) if isinstance(msg, HumanMessage)
        else SerializableAIMessage(content=msg.content)
        for msg in result["chat_history"]
    ]

    return QueryResponse(
        answer=result["answer"],
        chat_history=updated_history_serializable
    )


@app.get("/")
def read_root():
    return {"message": "Bug Bounty Chatbot is running"}
