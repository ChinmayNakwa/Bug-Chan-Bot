import os
import time
from dotenv import load_dotenv

from app.graph import app_graph  
from uagents_adapter import LangchainRegisterTool, cleanup_uagent

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("A GOOGLE_API_KEY must be set in your .env file.")

AGENTVERSE_API_KEY = os.getenv("AGENTVERSE_API_KEY")
if not AGENTVERSE_API_KEY:
    raise ValueError("An AGENTVERSE_API_KEY must be set in your .env file.")


def bug_bounty_agent_func(query: str) -> str:
    """
    This function serves as the interface between the uagents-adapter and our LangGraph agent.
    """
    
    if not isinstance(query, str):
        return "Error: Input must be a string."

    try:
        inputs = {"query": query, "chat_history": []}
        
        result = app_graph.invoke(inputs)
        
        final_answer = result.get('answer', "No valid answer was generated.")
        return final_answer

    except Exception as e:
        print(f"An error occurred during agent execution: {e}")
        return "Sorry, I encountered an error while processing your request."

print("🚀 Starting the Bug Bounty Agent...")

AGENT_NAME = "bug_bounty_rag_agent"

tool = LangchainRegisterTool()
agent_info = tool.invoke(
    {
        "agent_obj": bug_bounty_agent_func,
        "name": AGENT_NAME,
        "port": 8000, 
        "description": "A RAG-powered agent for answering bug bounty and website questions.",
        "api_token": AGENTVERSE_API_KEY,
        "mailbox": True  
    }
)

print(f"✅ Agent '{AGENT_NAME}' registered successfully!")
print(f"Agent Info: {agent_info}")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print(f"\n🛑 Shutting down '{AGENT_NAME}'...")
    cleanup_uagent(AGENT_NAME)
    print("✅ Agent stopped gracefully.")