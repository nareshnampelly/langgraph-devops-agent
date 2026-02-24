# LangGraph DevOps Agent

This project implements a stateful DevOps troubleshooting agent using LangGraph for workflow orchestration and LangSmith for evaluation.

## Features

- Multi-step workflow (retrieve → answer → judge → route)
- Explicit state management
- Conditional retry logic
- LLM-as-judge correctness scoring
- Evaluation via LangSmith SDK and UI

## How to Run

1. Create a virtual environment
2. Install dependencies:
   pip install -r requirements.txt
3. Set environment variables:
   OPENAI_API_KEY
   LANGSMITH_API_KEY
4. Run agent:
   python agent.py
5. Run evaluation:
   python eval_sdk.py