# Intelligent Bug Bounty RAG Chatbot

This project is an advanced, multi-agent chatbot designed for BugChan. It uses a stateful graph architecture built with LangGraph to intelligently route user queries to the correct specialized agent. Each agent is powered by Retrieval-Augmented Generation (RAG) to provide accurate answers from a dedicated knowledge base.

The agent is designed for deployment on [AgentVerse](https://agentverse.ai/) using the `uagents-adapter` library.

## ✨ Features

-   **Dual-Agent Architecture**:
    -   **Agent 1 (General Hacking)**: Answers conceptual questions about bug bounties, vulnerabilities (XSS, IDOR, etc.), and hacking terminology.
    -   **Agent 2 (Website Guide)**: Provides guidance on how to use BugChan, such as instructions for submitting reports, or navigating pages.
-   **Intelligent Routing**: A router powered by a Large Language Model (LLM) analyzes incoming queries and directs them to the most appropriate agent.
-   **Retrieval-Augmented Generation (RAG)**: Both agents use RAG to pull information from their respective knowledge bases (`app/knowledge.py`), ensuring answers are accurate and easily updatable without changing code.
-   **Local & Free Embeddings**: Utilizes a high-performance, self-hosted Hugging Face model for creating text embeddings, eliminating the need for paid embedding APIs.
-   **Persistent Vector Store**: Uses FAISS to create and save a local vector database, so knowledge is only processed once, making subsequent startups instantaneous.
-   **Answer Reflection**: An LLM-based quality check reviews the generated answer for relevance and helpfulness before finalizing the response.
-   **Ready for AgentVerse**: Built from the ground up to be deployed on AgentVerse using the `uagents-adapter` framework.

## 🏛️ Architecture Flow

1.  **Query Input**: The user sends a query to the agent.
2.  **Decision Node (Router)**: An LLM classifies the query as either `general_hacking` or `website_specific`.
3.  **Agent Execution**: The query is passed to the selected agent (Agent 1 or Agent 2).
4.  **RAG Process**: The agent retrieves relevant documents from its FAISS vector store and uses an LLM to generate a contextual answer.
5.  **Reflection Node (Quality Check)**: A final LLM call checks if the generated answer is relevant and helpful.
6.  **Final Answer**: If the answer passes the quality check, it is returned to the user.

## 🛠️ Tech Stack

-   **Core Framework**: LangChain & LangGraph
-   **LLM**: Google Gemini (`gemini-2.5-flash`)
-   **Embeddings**: Hugging Face Sentence Transformers
-   **Vector Store**: FAISS (Facebook AI Similarity Search)
-   **Deployment**: AgentVerse (`uagents-adapter`)

## 🚀 Getting Started

Follow these instructions to set up and run the agent on your local machine.

### 1. Prerequisites

-   Python 3.11+
-   Git

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/bug-bounty-chatbot.git
cd bug-bounty-chatbot