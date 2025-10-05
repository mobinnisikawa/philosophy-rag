import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader
import os

# ページ設定
st.set_page_config(page_title="哲学RAG", layout="wide")
st.title("📘 哲学RAGシステム (Streamlit版)")

# --- PDF アップロード ---
uploaded_files = st.file_uploader(
    "PDFをアップロードしてください（複数可）",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    all_docs = []
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        text = "".join(p.extract_text() or "" for p in reader.pages)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = [
            Document(page_content=chunk, metadata={"source": uploaded_file.name})
            for chunk in splitter.split_text(text)
        ]
        all_docs.extend(docs)

    st.success(f"✅ 全PDFの合計チャンク数: {len(all_docs)}")

    # --- ベクトルDB作成 ---
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(all_docs, emb)
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20})

    # --- QA ---
    query = st.text_input("質問を入力してください")
    if query:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([d.page_content for d in docs])

        prompt = f"次の資料に基づいて質問に答えてください。\n\n資料:\n{context}\n\n質問: {query}\n答え:"
        answer = llm.predict(prompt)

        st.subheader("🔎 回答")
        st.write(answer)

        # 参照元表示
        with st.expander("参照したドキュメント"):
            for d in docs:
                st.markdown(f"- **{d.metadata['source']}**: {d.page_content[:200]}...")
else:
    st.info("左のボックスからPDFファイルをアップロードしてください。")

