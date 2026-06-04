"""Verification agent: fact-check the draft against the evidence. Acts as the
loop-back gate via structured output."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..clients import reasoning_llm
from ..state import AgentState

_PROMPT = (
    "You are a strict fact-checker. Compare the draft answer against the evidence. "
    "Identify any claim in the draft that is NOT supported by the evidence. "
    "Set supported=true ONLY if every substantive claim is backed by the evidence.\n\n"
    "Question: {question}\n\n"
    "Evidence:\n{evidence}\n\n"
    "Draft answer:\n{draft}"
)


class VerificationModel(BaseModel):
    supported: bool = False
    unsupported_claims: list[str] = Field(default_factory=list)
    rationale: str = ""


def verification_node(state: AgentState) -> dict:
    llm = reasoning_llm().with_structured_output(VerificationModel)
    result: VerificationModel = llm.invoke(
        _PROMPT.format(
            question=state["question"],
            evidence=state.get("evidence", ""),
            draft=state.get("draft_answer", ""),
        )
    )
    return {
        "verification": result.model_dump(),
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
