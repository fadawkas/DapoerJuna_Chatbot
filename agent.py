import json
import re
import random
from typing import List, TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from retriever import retriever
from memory import memory, remember
from tools import TOOLS

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    convert_system_message_to_human=True
)

class ChatState(TypedDict):
    messages: Annotated[List[str], ...]
    steps: int
    docs: str | None
    rewritten: str | None
    route: str | None
    error: str | None
    attitude: str


def juna_style(att: str) -> str:
    if att == "random":
        att = random.choice(["baik", "galak"])
    return (
        "Kamu menjawab seperti Gordon Ramsay: tegas, sinis, namun tetap sopan."
        if att in ("galak", "mean")
        else "Kamu menjawab ramah, antusias, dan suportif."
    )


SYSTEM_BASE = (
    "Kamu adalah chef virtual bernama Juna yang ahli resep masakan Indonesia.\n"
    "Gunakan hanya data dari KONTEKS RESEP yang diberikan di database vector.\n"
    "Jika perlu menggunakan tool, gunakan hanya tool yang disediakan di bawah ini:\n"
    "- retrieve_recipe(query: str)\n"
    "- get_recipe(query: str)\n"
    "- filter_by_category(recipes: str, category: str)\n"
    "- filter_by_difficulty(recipes: str, difficulty: str)\n"
    "- filter_by_weight(recipes: str, meal_weight: str)\n"
    "- filter_by_ingredients(recipes: str, ingredients: str)\n"
    "- get_recipe_details(selection: str, recipes: str | None = None)\n"
    "- get_most_loved(top_n: int = 5, recipes: str | None = None)\n"
    "- set_juna_attitude(attitude: str = 'baik')\n"
    "\n"
    "Format pemanggilan tool yang benar:\n"
    "<tool>CALL_nama_tool {\"arg1\": \"value1\", \"arg2\": \"value2\"}</tool>\n"
    "\n"
    "Jangan gunakan tool lain di luar daftar ini! Jika tidak perlu tool, jawab langsung."
)



def safe_tool(name: str, args: dict):
    try:
        return TOOLS[name](**args)
    except Exception as e:
        raise RuntimeError(f"Tool `{name}` gagal: {e}")


def build_agent():
    g = StateGraph(ChatState)

    # ────── Nodes ──────────────────────
    def rewrite_node(state: ChatState) -> ChatState:
        user_last = state["messages"][-1]
        prompt = (
            juna_style(state["attitude"]) + "\n" + SYSTEM_BASE +
            "\n\nRewrite pertanyaan berikut agar cocok untuk pencarian resep:\n"
            f"{user_last}\n\nRewritten:"
        )
        rew = llm.invoke(prompt).content.strip()
        state["rewritten"] = rew
        state["messages"].append(f"[rewritten] {rew}")
        remember("ai", f"[rewritten] {rew}")
        return state

    def retrieve_node(state: ChatState) -> ChatState:
        docs = retriever.get_relevant_documents(state["rewritten"])
        state["docs"] = "\n\n".join(d.page_content for d in docs)
        return state

    def router_node(state: ChatState) -> ChatState:
        last = state["messages"][-1].lower()
        if last.startswith("user:"):
            last = last[5:].strip()

        if "juna" in last and any(w in last for w in
            ["mean", "galak", "kejam", "garang", "nice", "random", "attitude", "sikap"]):
            state["route"] = "att_change"
        elif any(k in last for k in ["paling disukai", "most loved", "favorit"]):
            state["route"] = "by_loves"
        elif any(k in last for k in ["mudah", "gampang", "sedang", "cukup rumit", "sulit", "ribet", "susah", "cepat", "lama"]):
            state["route"] = "by_difficulty"
        elif any(k in last for k in ["vegan", "non vegan", "tanpa daging"]):
            state["route"] = "by_diet"
        elif any(k in last for k in ["ringan", "berat"]):
            state["route"] = "by_weight"
        elif any(k in last for k in ["ayam", "sapi", "ikan", "kambing", "udang", "telur", "bahan", "punya", "ada"]):
            state["route"] = "by_ingredients"
        else:
            state["route"] = "rag_answer"
        return state

    def att_set_node(state: ChatState) -> ChatState:
        user_msg = state["messages"][-1].lower()
        match = re.search(r"\b(baik|galak|mean|random)\b", user_msg)
        new_att = match.group(1) if match else "baik"
        state["attitude"] = new_att
        msg = f"Sikap Juna di-set ke '{new_att}'."
        state["messages"].append(msg)
        remember("ai", msg)
        return state

    def decide_node(state: ChatState) -> ChatState:
        hist = memory.load_memory_variables({})["history"]
        prompt = (
            juna_style(state["attitude"]) + "\n" + SYSTEM_BASE +
            f"\n\n{hist}\n" + "\n".join(state["messages"]) + "\n\nAssistant:"
            "Jika hasilnya lebih dari satu resep, berikan hanya daftar judul resepnya (maksimal 5) tanpa detail. Minta user memilih salah satu untuk melihat detailnya."
        )
        draft = llm.invoke(prompt).content.strip()
        state["messages"].append(draft)
        remember("ai", draft)
        state["steps"] += 1
        return state

    def tool_node(state: ChatState) -> ChatState:
        call = state["messages"][-1]
        name_m = re.search(r"CALL_(\w+)", call)
        json_m = re.search(r"\{.*?\}", call, flags=re.S)
        if not (name_m and json_m):
            state["error"] = "Tool format salah."
            return state
        name = name_m.group(1)
        args = json.loads(json_m.group(0))
        try:
            result = TOOLS[name].invoke(args)
            state["messages"].append(result)
            remember("ai", result)
        except Exception as e:
            state["error"] = f"Tool `{name}` gagal: {e}"
        return state


    def synth_node(state: ChatState) -> ChatState:
        prompt = (
            juna_style(state["attitude"]) + "\n" + SYSTEM_BASE +
            f"\n\nDocs:\n{state['docs']}\n\nPertanyaan: {state['rewritten']}\n\nJawaban ringkas (Bahasa Indonesia):"
        )
        ans = llm.invoke(prompt).content.strip()
        state["messages"].append(ans)
        remember("ai", ans)
        return state

    def error_node(state: ChatState) -> ChatState:
        msg = state.get("error", "Maaf, terjadi kesalahan.")
        state["messages"].append(msg)
        remember("ai", msg)
        return state

    # ────── Graph ──────────────────────
    g.add_node("rewrite", rewrite_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("router", router_node)
    g.add_node("att_set", att_set_node)
    g.add_node("decide", decide_node)
    g.add_node("tool", tool_node)
    g.add_node("rag_ans", synth_node)
    g.add_node("error_llm", error_node)

    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "router")

    g.add_conditional_edges("router", lambda s: s["route"], {
        "att_change": "att_set",
        "by_loves": "decide",
        "by_diet": "decide",
        "by_ingredients": "decide",
        "by_weight": "decide",
        "by_difficulty": "decide",
        "rag_answer": "rag_ans"
    })

    g.add_conditional_edges("decide", lambda s: "tool" if "CALL_" in s["messages"][-1] else "end", {
        "tool": "tool",
        "end": END
    })

    g.add_conditional_edges("tool", lambda s: "error_llm" if s.get("error") else "decide", {
        "decide": "decide",
        "error_llm": "error_llm"
    })

    g.add_edge("att_set", END)
    g.add_edge("rag_ans", END)
    g.add_edge("error_llm", END)

    g.set_entry_point("rewrite")
    return g.compile()
