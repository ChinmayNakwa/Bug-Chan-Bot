import os
from dotenv import load_dotenv
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents import agent_1_chain, agent_2_chain

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("A GOOGLE_API_KEY must be set in your .env file.")

llm_router = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

class AgentState(TypedDict):
    query: str
    answer: str
    chat_history: List[AnyMessage]

def agent_1_node(state: AgentState) -> dict:
    """Node for the general bug bounty agent."""
    print("--- EXECUTING AGENT 1: GENERAL HACKING ---")
    result = agent_1_chain.invoke({
        "input": state['query'], 
        "chat_history": state.get('chat_history', [])
    })
    return {"answer": result['answer']}

def agent_2_node(state: AgentState) -> dict:
    """Node for the website guide agent."""
    print("--- EXECUTING AGENT 2: WEBSITE GUIDE ---")
    result = agent_2_chain.invoke({
        "input": state['query'], 
        "chat_history": state.get('chat_history', [])
    })
    return {"answer": result['answer']}

def update_history_node(state: AgentState) -> dict:
    """Updates the chat history with the latest query and answer."""
    print("--- UPDATING CONVERSATIONAL MEMORY ---")
    history = state.get('chat_history', [])
    # Keep last 10 messages (5 exchanges)
    if len(history) > 10:
        history = history[-10:]
    history.append(HumanMessage(content=state['query']))
    history.append(SystemMessage(content=state['answer']))
    return {"chat_history": history}

def decision_node(state: AgentState) -> str:
    """Uses an LLM to decide which agent should handle the query."""
    print("--- ROUTER: DECIDING PATH ---")
    prompt = f"""
    You are a router. Your task is to classify the user's query into one of two categories based on its content:
    1. 'general_hacking': For questions about general bug bounty concepts, vulnerabilities, hacking techniques, PoCs, etc.
    2. 'website_specific': For questions about how to use BugChan, such as logging in, submitting reports, finding pages, or managing an account.

    Based on the user's query, which category is more appropriate?
    User Query: "{state['query']}"

    Return only the single word 'general_hacking' or 'website_specific'.
    """
    response = llm_router.invoke(prompt)
    decision = response.content.strip().lower()

    if "website_specific" in decision:
        print("Router decision: 'website_specific'")
        return "agent_2"
    else:
        print("Router decision: 'general_hacking'")
        return "agent_1"

def reflect_and_decide(state: AgentState) -> str:
    """
    Checks the quality of the generated answer and decides whether to continue or end.
    This function acts as a conditional edge.
    """
    print("--- REFLECTOR: CHECKING ANSWER QUALITY ---")
    prompt = f"""
    You are a quality checker. Your task is to determine if the provided answer is a relevant and helpful response to the user's query.
    The answer should NOT be a refusal like "I don't have enough information."

    User Query: "{state['query']}"
    Generated Answer: "{state['answer']}"

    Is the answer relevant and sufficient? Answer with only the single word 'yes' or 'no'.
    """
    response = llm_router.invoke(prompt)
    decision = response.content.strip().lower()

    if "yes" in decision:
        print("Reflector decision: 'continue' (Answer is good)")
        return "continue"
    else:
        print("Reflector decision: 'stop' (Answer is not good)")
        return "stop"
    
workflow = StateGraph(AgentState)

workflow.add_node("agent_1", agent_1_node)
workflow.add_node("agent_2", agent_2_node)
workflow.add_node("update_history", update_history_node)

workflow.add_conditional_edges(
    "__start__",
    decision_node,
    {
        "agent_1": "agent_1",
        "agent_2": "agent_2",
    }
)

workflow.add_conditional_edges(
    "agent_1",
    reflect_and_decide,
    {
        "continue": "update_history",
        "stop": END,
    }
)

workflow.add_conditional_edges(
    "agent_2",
    reflect_and_decide,
    {
        "continue": "update_history",
        "stop": END,
    }
)

workflow.add_edge("update_history", END)

memory_store = MemorySaver()
app_graph = workflow.compile(checkpointer=None)

# Example of invoking with session-based memory
# Each session_id corresponds to a user or conversation thread
# 
# First message in a conversation:
# result = app_graph.invoke(
#     {"query": "How do I find SQL injection vulnerabilities?", "answer": "", "chat_history": []},
#     config={"configurable": {"thread_id": "session_123"}}
# )
# print(result['answer'])
#
# Follow-up message in the same conversation (memory will be automatically loaded):
# result = app_graph.invoke(
#     {"query": "Can you give me an example?", "answer": "", "chat_history": []},
#     config={"configurable": {"thread_id": "session_123"}}
# )
# print(result['answer'])