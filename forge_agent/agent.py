"""LangGraph chat agent that loads personality from AgentHub."""

from __future__ import annotations

import os
from typing import Annotated, Any, TypedDict

import httpx
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

AGENT_ID = os.environ.get("AGENT_ID", "").strip()
AGENTHUB_URL = os.environ.get("AGENTHUB_URL", "").rstrip("/")
RUNTIME_SECRET = os.environ.get("RUNTIME_SECRET", "").strip()

llm = ChatOpenAI(
    api_key=os.environ["HARNESS_INFERENCE_API_KEY"],
    base_url=os.environ.get("HARNESS_INFERENCE_BASE_URL", "https://inference.do-ai.run/v1").rstrip(
        "/"
    ),
    model=os.environ["HARNESS_INFERENCE_MODEL"],
    temperature=0.2,
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _runtime_langchain_config(
    name: str, description: str, langchain_config: dict[str, Any] | None
) -> dict[str, Any]:
    """Mirror AgentHub runtime_langchain_config for system_prompt assembly."""
    cfg = dict(langchain_config or {})
    name = (name or "").strip()
    name_line = f"Your name is {name}." if name else ""
    existing = str(cfg.get("system_prompt") or "").strip()
    if existing:
        if name_line and not existing.startswith("Your name is "):
            cfg["system_prompt"] = f"{name_line}\n\n{existing}"
        return cfg

    parts: list[str] = []
    if name_line:
        parts.append(name_line)
    desc = (description or "").strip()
    if desc:
        parts.append(f"Description: {desc}")
    transcript = str(cfg.get("builder_transcript") or "").strip()
    if transcript:
        parts.append("The builder conversation below defines your job. Follow it.\n\n" + transcript)
    cfg["system_prompt"] = "\n\n".join(parts) if parts else "You are a helpful assistant."
    return cfg


def fetch_system_prompt() -> str:
    if not AGENTHUB_URL or not AGENT_ID or not RUNTIME_SECRET:
        return "You are a helpful assistant."
    url = f"{AGENTHUB_URL}/api/v1/internal/agents/{AGENT_ID}/config"
    with httpx.Client(timeout=30.0) as client:
        r = client.get(url, headers={"Authorization": f"Bearer {RUNTIME_SECRET}"})
        r.raise_for_status()
        data = r.json()
    cfg = _runtime_langchain_config(
        str(data.get("name") or ""),
        str(data.get("description") or ""),
        data.get("langchain_config") if isinstance(data.get("langchain_config"), dict) else {},
    )
    return str(cfg.get("system_prompt") or "You are a helpful assistant.")


def _with_system(messages: list[BaseMessage], system_prompt: str) -> list[BaseMessage]:
    if messages and isinstance(messages[0], SystemMessage):
        return [SystemMessage(content=system_prompt), *messages[1:]]
    return [SystemMessage(content=system_prompt), *messages]


def respond(state: AgentState) -> dict[str, list[BaseMessage]]:
    system_prompt = fetch_system_prompt()
    reply = llm.invoke(_with_system(list(state["messages"]), system_prompt))
    return {"messages": [reply]}


workflow = StateGraph(AgentState)
workflow.add_node("respond", respond)
workflow.add_edge(START, "respond")
workflow.add_edge("respond", END)

agent = workflow.compile()
