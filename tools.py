import re
import pandas as pd
from langchain.tools import Tool, tool,  StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI

CSV_PATH = "database/df_resep_cleaned.csv"
df = pd.read_csv(CSV_PATH)
df = (
    df.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
    .rename(columns={"difficulty_level": "difficulty"})
)
non_vegan = {"ayam", "kambing", "sapi", "ikan", "udang", "telur"}
df["diet"] = df.category.apply(lambda c: "non vegan" if c in non_vegan else "vegan")
df["meal_weight"] = df.total_ingredients.apply(lambda x: "ringan" if x <= 8 else "berat")


def build_block(r) -> str:
    return (
        f"Judul: {r.title}  (Loved: {r.loves})\n"
        f"Kategori: {r.category}\n"
        f"Diet: {r.diet}\n"
        f"Difficulty: {r.difficulty}\n"
        f"Bahan: {r.ingredients}\n"
        f"Berat: {r.meal_weight}\n"
        f"Loves: {r.loves}\n"
        f"Langkah:\n{r.steps}"
    )


@tool
def retrieve_recipe(query: str, k: int = 4) -> str:
    """RAG – k resep paling relevan untuk `query`."""
    from retriever import retriever
    docs = retriever.invoke(query)[:k]
    return "\n\n".join(d.page_content for d in docs)


@tool
def get_recipe(query: str, k: int = 4) -> str:
    """Alias dari `retrieve_recipe`."""
    return retrieve_recipe.invoke({"query": query, "k": k})

def _filter_by_category(recipes: str, category: str) -> str:
    """Filter blok resep yang memiliki `Kategori:` sesuai argumen."""
    cat = category.lower()
    return "\n\n".join(
        r for r in recipes.split("\n\n") if f"kategori: {cat}" in r.lower()
    )

def _filter_by_weight(recipes: str, meal_weight: str) -> str:
    """Filter blok resep berdasarkan meal_weight: 'ringan' atau 'berat'."""
    key = meal_weight.lower().split()[0]
    return "\n\n".join(r for r in recipes.split("\n\n") if key in r.lower())

def _filter_by_difficulty(recipes: str, difficulty: str) -> str:
    """Filter blok resep: 'mudah', 'sedang', 'cukup rumit', 'sulit'."""
    diff = difficulty.lower()
    return "\n\n".join(r for r in recipes.split("\n\n") if diff in r.lower())

def _filter_by_ingredients(ingredients: str, recipes: str | None = None) -> str:
    """Kembalikan resep yang semua bahannya ada di input (comma-separated)."""
    from streamlit import session_state as st_session
    recipes = recipes or st_session.last_recipes_blob

    want = {i.strip().lower() for i in ingredients.split(",")}
    out = []
    for blk in recipes.split("\n\n"):
        m = re.search(r"Bahan:(.*)", blk, flags=re.S)
        have = {i.strip().lower() for i in m.group(1).split(",")} if m else set()
        if want.issubset(have):
            out.append(blk)
    return "\n\n".join(out)

def get_most_loved(dummy: str = "") -> str:
    """
    Ambil resep paling disukai. Dapat dipanggil tanpa argumen.
    """
    top_df = df.sort_values("loves", ascending=False).head(5)
    return "\n\n".join(build_block(r) for r in top_df.itertuples(index=False))

@tool
def set_juna_attitude(attitude: str = "baik") -> str:
    """Ubah sikap Juna ke 'baik', 'galak/mean', atau 'random'."""
    return attitude.lower()


def _pick(blocks: list[str], sel: str) -> str | None:
    sel = sel.strip().lower()
    if sel.isdigit():
        idx = int(sel) - 1
        return blocks[idx] if 0 <= idx < len(blocks) else None
    for b in blocks:
        if sel in b.lower():
            return b
    return None

def get_recipe_details(selection: str, recipes: str | None = None) -> str:
    """
    Ambil 1 resep dari kumpulan `recipes`.

    - `selection`: bisa berupa nomor (1-based) atau sebagian judul resep.
    - Jika `recipes` tidak diberikan, akan menggunakan hasil pencarian terakhir.
    - Jika tidak ketemu, akan mengembalikan string kosong.
    """
    from streamlit import session_state as st_session

    recipes = recipes or st_session.get("last_recipes_blob", "")
    blocks = [b for b in recipes.split("\n\n") if b.strip()]
    pick = _pick(blocks, selection)
    return pick or ""


TOOLS = {
    "retrieve_recipe": Tool.from_function(
        func=retrieve_recipe,
        name="retrieve_recipe",
        description="RAG – k resep paling relevan untuk query."
    ),
    "get_recipe": Tool.from_function(
        func=get_recipe,
        name="get_recipe",
        description="Alias dari retrieve_recipe."
    ),
    "filter_by_category": StructuredTool.from_function(
        func=_filter_by_category,
        name="filter_by_category",
        description="Filter blok resep berdasarkan kategori (contoh: 'ayam', 'sapi', dll)."
    ),
    "filter_by_weight": StructuredTool.from_function(
        func=_filter_by_weight,
        name="filter_by_weight",
        description="Filter blok resep berdasarkan berat makanan: 'ringan' atau 'berat'."
    ),
    "filter_by_difficulty": StructuredTool.from_function(
        func=_filter_by_difficulty,
        name="filter_by_difficulty",
        description="Filter blok resep berdasarkan tingkat kesulitan: 'mudah', 'sedang', 'cukup rumit', atau 'sulit'."
    ),
    "filter_by_ingredients": StructuredTool.from_function(
        func=_filter_by_ingredients,
        name="filter_by_ingredients",
        description="Filter resep yang semua bahannya ada dalam daftar bahan pengguna."
    ),
    "get_most_loved": StructuredTool.from_function(
        func=get_most_loved,
        name="get_most_loved",
        description="Ambil top-n resep yang paling banyak disukai."
    ),
    "set_juna_attitude": Tool.from_function(
        func=set_juna_attitude,
        name="set_juna_attitude",
        description="Ubah sikap Juna ke 'baik', 'galak', atau 'random'."
    ),
    "get_recipe_details": StructuredTool.from_function(
        func=get_recipe_details,
        name="get_recipe_details",
        description="Ambil 1 resep dari hasil pencarian sebelumnya berdasarkan nomor atau judul parsial.",
    )
}
