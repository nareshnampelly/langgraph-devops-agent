import os
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from agent import build_graph
from langchain_openai import ChatOpenAI

# Initialize LangSmith client
client = Client()

DATASET_NAME = "devops_agent_eval"

# Build your LangGraph app
app = build_graph()


def target(inputs: dict) -> dict:
    """
    This function runs your LangGraph agent
    for each dataset example.
    """
    init = {
        "question": inputs["question"],
        "docs": [],
        "draft": "",
        "score": 0.0,
        "retries": 0
    }

    out = app.invoke(init)

    # simple retry logic
    if out["score"] < 0.7 and out["retries"] < 1:
        out["retries"] = 1
        out = app.invoke(out)

    return {
        "answer": out["draft"],
        "confidence": out["score"]
    }


def correctness_evaluator(run, example):
    """
    LLM-as-judge evaluator.
    Scores semantic correctness, not literal match.
    """

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
    )

    expected = example.outputs.get("expected", "")
    answer = run.outputs.get("answer", "")

    prompt = f"""
You are evaluating answer quality.

Score from 0.0 to 1.0 based on:
- Does the answer correctly address the same core troubleshooting concepts as the expected answer?
- Minor wording differences should NOT reduce score.
- Additional helpful detail should NOT reduce score.
- Only reduce score if the answer is incorrect, misleading, or missing key concepts.

Return ONLY a number between 0.0 and 1.0.

Expected Answer:
{expected}

Agent Answer:
{answer}
"""

    score_text = llm.invoke(prompt).content.strip()

    try:
        score = float(score_text)
    except Exception:
        score = 0.5

    return {"key": "correctness", "score": score}


if __name__ == "__main__":
    results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator],
        experiment_prefix="devops-agent",
    )

    print("Evaluation started. Check LangSmith Experiments.")