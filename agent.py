import os
from typing import TypedDict, List

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# -----------------------------
# 1) state that flows through the graph
# -----------------------------
class AgentState(TypedDict):
    question: str
    docs: List[str]
    draft: str
    score: float
    retries: int


# -----------------------------
# 2) Simple "retriever" from kb.md
#    (Not vector DB; intentionally simple for reliability)
# -----------------------------
def retrieve_docs(state: AgentState) -> AgentState:
    question = state["question"].lower()
    with open("kb.md", "r", encoding="utf-8") as f:
        kb = f.read()

    # naive chunking by section
    sections = kb.split("\n## ")
    hits = []

    # simple keyword matching
    keywords = []
    if "ecs" in question or "rds" in question or "database" in question:
        keywords += ["ECS", "RDS", "connect"]
    if "peering" in question or "vpc" in question:
        keywords += ["peering", "route", "VPC"]
    if "crashloop" in question:
        keywords += ["CrashLoopBackOff", "probe", "secret"]

    for sec in sections:
        # put section header back for readability
        sec_text = ("## " + sec) if not sec.startswith("#") else sec
        if any(k.lower() in sec_text.lower() for k in keywords) or any(w in sec_text.lower() for w in question.split()):
            hits.append(sec_text.strip())

    # keep it small
    hits = hits[:3] if hits else [kb[:800]]

    state["docs"] = hits
    return state


# -----------------------------
# 3) Draft answer node
# -----------------------------
def generate_answer(state: AgentState) -> AgentState:
    llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
)
    context = "\n\n".join(state["docs"])
    prompt = f"""
You are a DevOps troubleshooting assistant.
Answer using ONLY the provided context when possible.
If context is insufficient, ask 1-2 precise follow-up questions.

Question: {state['question']}

Context:
{context}

Return:
- Root cause possibilities (bullets)
- Recommended checks (bullets)
- Suggested next command(s) if relevant
"""

    resp = llm.invoke(prompt)
    state["draft"] = resp.content
    return state


# -----------------------------
# 4) Judge node: score 0-1 confidence
# -----------------------------
def judge_answer(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    context = "\n\n".join(state["docs"])
    prompt = f"""
You are grading an assistant answer.
Score confidence from 0.0 to 1.0 based on:
- Uses the provided context (no wild hallucinations)
- Answer is actionable and matches the question
- If context is thin, it asks good follow-up questions

Return ONLY a number between 0.0 and 1.0.

Question: {state['question']}

Context:
{context}

Answer:
{state['draft']}
"""
    resp = llm.invoke(prompt).content.strip()
    try:
        state["score"] = float(resp)
    except Exception:
        state["score"] = 0.5
    return state


# -----------------------------
# 5) Router: retry once if low score
# -----------------------------
def route(state: AgentState) -> str:
    if state["score"] < 0.7 and state["retries"] < 1:
        return "retry"
    return "final"


# -----------------------------
# 6) Build graph
# -----------------------------
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve_docs)
    g.add_node("answer", generate_answer)
    g.add_node("judge", judge_answer)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "judge")

    g.add_conditional_edges("judge", route, {
        "retry": "retrieve",
        "final": END,
    })

    return g.compile()


if __name__ == "__main__":
    app = build_graph()

    q = input("Ask a DevOps question: ").strip()
    init: AgentState = {
        "question": q,
        "docs": [],
        "draft": "",
        "score": 0.0,
        "retries": 0
    }

    # run; on retry we increment retries
    # (LangGraph state mutation is simple here; we update after each run step)
    result = app.invoke(init)

    # If it routed to retry, our state didn't increment retries automatically.
    # Simplest approach: if score low, run once more with retries=1
    if result["score"] < 0.7 and result["retries"] < 1:
        result["retries"] = 1
        result = app.invoke(result)

    print("\n--- FINAL ANSWER ---\n")
    print(result["draft"])
    print(f"\n[confidence={result['score']:.2f}, retries={result['retries']}]")
