from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

import re

from .llm import extractive_fallback, generate_answer
from .retrieve import (
    filter_results,
    format_results,
    load_reranker,
    load_retrieval_model,
    retrieve_chunks,
)


class AgentState(TypedDict, total=False):
    question: str
    interaction_intent: str
    retrieval_query: str
    is_follow_up: bool
    needs_broad_search: bool
    retrieve_top_k: int
    use_reranker: bool
    retrieved_chunks: list[dict]
    answer: str
    generation_model: str
    agent_trace: list[str]
    chat_history: list[dict]


def classify_question(state: AgentState) -> AgentState:
    question_text = state["question"].strip()
    question = question_text.lower()
    question_words = question_text.split()
    history = state.get("chat_history", [])
    follow_up_markers = ["and ", "what about", "how about", "that", "those", "them", "it", "also"]
    broad_question_starters = (
        "what is",
        "what are",
        "what does",
        "how does",
        "how do",
        "why does",
        "why do",
        "describe",
        "explain",
        "summarize",
        "tell me",
    )
    smalltalk_phrases = {
        "ok",
        "okay",
        "cool",
        "thanks",
        "thank you",
        "thx",
        "got it",
        "nice",
        "great",
        "awesome",
        "understood",
        "makes sense",
    }
    interaction_intent = "question"

    if question in smalltalk_phrases or (
        len(question.split()) <= 4
        and any(phrase in question for phrase in smalltalk_phrases)
    ):
        interaction_intent = "smalltalk"

    is_broad_question = (
        question.startswith(broad_question_starters)
        or len(question_words) >= 9
    )
    needs_broad_search = interaction_intent == "question" and is_broad_question
    is_follow_up = interaction_intent == "question" and bool(history) and (
        len(question.split()) <= 8 or any(marker in question for marker in follow_up_markers)
    )

    trace = state.get("agent_trace", []) + [
        f"classify_question: intent={interaction_intent}, broad_search={needs_broad_search}, follow_up={is_follow_up}"
    ]

    return {
        "interaction_intent": interaction_intent,
        "is_follow_up": is_follow_up,
        "needs_broad_search": needs_broad_search,
        "retrieve_top_k": 4,
        "use_reranker": True,
        "retrieval_query": state["question"],
        "agent_trace": trace,
    }


def route_after_classification(state: AgentState) -> Literal["respond_smalltalk", "rewrite_query"]:
    if state.get("interaction_intent") == "smalltalk":
        return "respond_smalltalk"
    return "rewrite_query"


def respond_smalltalk(state: AgentState) -> AgentState:
    question = state["question"].strip().lower()

    if "thank" in question or "thx" in question:
        answer = "You're welcome. Ask the next question when you're ready."
    elif question in {"ok", "okay", "cool", "nice", "great", "awesome"}:
        answer = "Understood. Ask the next question when you're ready."
    else:
        answer = "Understood."

    trace = state.get("agent_trace", []) + ["respond_smalltalk: no_retrieval"]
    return {
        "answer": answer,
        "generation_model": "smalltalk-rule",
        "retrieved_chunks": [],
        "agent_trace": trace,
    }


def rewrite_query(state: AgentState) -> AgentState:
    question = state["question"]
    history = state.get("chat_history", [])

    if not state.get("is_follow_up") or not history:
        trace = state.get("agent_trace", []) + ["rewrite_query: original"]
        return {"retrieval_query": question, "agent_trace": trace}

    last_user_messages = [item["content"] for item in history if item.get("role") == "user"]
    if not last_user_messages:
        trace = state.get("agent_trace", []) + ["rewrite_query: original"]
        return {"retrieval_query": question, "agent_trace": trace}

    previous_question = last_user_messages[-1]
    retrieval_query = f"Previous question: {previous_question}\nFollow-up question: {question}"
    trace = state.get("agent_trace", []) + ["rewrite_query: contextualized_follow_up"]
    return {"retrieval_query": retrieval_query, "agent_trace": trace}


def retrieve_primary(state: AgentState) -> AgentState:
    model = load_retrieval_model()
    reranker = load_reranker() if state.get("use_reranker", True) else None

    results = retrieve_chunks(
        state.get("retrieval_query", state["question"]),
        model,
        top_k=state.get("retrieve_top_k", 4),
        use_reranker=state.get("use_reranker", True),
        reranker=reranker,
    )
    formatted = format_results(results)
    filtered = filter_results(formatted)

    trace = state.get("agent_trace", []) + [
        f"retrieve_primary: retrieved={len(filtered)}"
    ]

    return {
        "retrieved_chunks": filtered,
        "agent_trace": trace,
    }


def assess_retrieval(state: AgentState) -> AgentState:
    retrieved = state.get("retrieved_chunks", [])
    broad_search = state.get("needs_broad_search", False)

    should_retry = False
    if not retrieved:
        should_retry = True
    elif broad_search and len(retrieved) < 2:
        should_retry = True

    trace = state.get("agent_trace", []) + [f"assess_retrieval: retry={should_retry}"]
    return {"agent_trace": trace, "needs_broad_search": should_retry}


def route_after_assessment(state: AgentState) -> Literal["retrieve_broader", "generate_answer"]:
    if state.get("needs_broad_search", False):
        return "retrieve_broader"
    return "generate_answer"


def retrieve_broader(state: AgentState) -> AgentState:
    model = load_retrieval_model()
    reranker = load_reranker()

    results = retrieve_chunks(
        state.get("retrieval_query", state["question"]),
        model,
        top_k=6,
        use_reranker=True,
        reranker=reranker,
    )
    formatted = format_results(results)
    filtered = filter_results(formatted)

    trace = state.get("agent_trace", []) + [
        f"retrieve_broader: retrieved={len(filtered)}"
    ]

    return {
        "retrieved_chunks": filtered,
        "agent_trace": trace,
        "needs_broad_search": False,
    }


def generate_answer_node(state: AgentState) -> AgentState:
    answer, model_name = generate_answer(
        state["question"],
        state.get("retrieved_chunks", []),
    )
    trace = state.get("agent_trace", []) + [f"generate_answer: model={model_name}"]
    return {
        "answer": answer,
        "generation_model": model_name,
        "agent_trace": trace,
    }


def verify_answer_node(state: AgentState) -> AgentState:
    answer = state.get("answer", "").strip()
    question = state["question"].lower()
    retrieved = state.get("retrieved_chunks", [])

    query_terms = {
        token
        for token in re.findall(r"\b[a-zA-Z]{4,}\b", question)
        if token not in {"what", "does", "rules", "work", "described", "say"}
    }
    answer_lower = answer.lower()
    has_query_overlap = any(term in answer_lower for term in query_terms)
    too_short = len(answer.split()) < 8
    says_not_found = (
        "does not contain information" in answer_lower
        or "not found in the retrieved context" in answer_lower
    )

    if (
        too_short
        or says_not_found
        or (query_terms and not has_query_overlap)
    ):
        fallback_answer = extractive_fallback(state["question"], retrieved)
        trace = state.get("agent_trace", []) + [
            "verify_answer: fallback_to_extractive"
        ]
        return {
            "answer": fallback_answer,
            "generation_model": f"{state.get('generation_model', 'unknown')}+verified",
            "agent_trace": trace,
        }

    trace = state.get("agent_trace", []) + ["verify_answer: accepted"]
    return {"agent_trace": trace}


def build_graph():
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("classify_question", classify_question)
    graph_builder.add_node("respond_smalltalk", respond_smalltalk)
    graph_builder.add_node("rewrite_query", rewrite_query)
    graph_builder.add_node("retrieve_primary", retrieve_primary)
    graph_builder.add_node("assess_retrieval", assess_retrieval)
    graph_builder.add_node("retrieve_broader", retrieve_broader)
    graph_builder.add_node("generate_answer", generate_answer_node)
    graph_builder.add_node("verify_answer", verify_answer_node)

    graph_builder.add_edge(START, "classify_question")
    graph_builder.add_conditional_edges(
        "classify_question",
        route_after_classification,
        {
            "respond_smalltalk": "respond_smalltalk",
            "rewrite_query": "rewrite_query",
        },
    )
    graph_builder.add_edge("respond_smalltalk", END)
    graph_builder.add_edge("rewrite_query", "retrieve_primary")
    graph_builder.add_edge("retrieve_primary", "assess_retrieval")
    graph_builder.add_conditional_edges(
        "assess_retrieval",
        route_after_assessment,
        {
            "retrieve_broader": "retrieve_broader",
            "generate_answer": "generate_answer",
        },
    )
    graph_builder.add_edge("retrieve_broader", "generate_answer")
    graph_builder.add_edge("generate_answer", "verify_answer")
    graph_builder.add_edge("verify_answer", END)

    return graph_builder.compile()
