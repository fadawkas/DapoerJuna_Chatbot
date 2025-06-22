import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from tools import df, build_block

load_dotenv()

INDEX_DIR = "database/faiss_recipes_index"
CSV_PATH = "data/df_resep_cleaned.csv"

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def load_retriever() -> FAISS:
    if os.path.exists(INDEX_DIR):
        vs = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
    else:
        docs = [
            Document(
                page_content=build_block(r),
                metadata={"loves": int(r.loves)},
            )
            for r in df.itertuples(index=False)
        ]
        vs = FAISS.from_documents(docs, embedding_model)
        vs.save_local(INDEX_DIR)
    return vs.as_retriever(search_kwargs={"k": 4})


retriever = load_retriever()
