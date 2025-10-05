import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader
import os, json

# --- ページ設定 ---
st.set_page_config(page_title="哲学RAGシステム", layout="wide")
st.title("📚 哲学RAGシステム (Streamlit版)")

# --- 履歴ファイル ---
HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# 履歴読み込み
history = load_history()

# --- サイドバーに履歴表示 ---
st.sidebar.header("💬 チャット履歴")
if history:
    for h in history[-5:]:  # 最新5件だけ表示
        st.sidebar.write(f"Q: {h['question']}")
        st.sidebar.caption(f"A: {h['answer'][:80]}...")
    if st.sidebar.button("全履歴を表示する"):
        for h in history:
            st.sidebar.write(f"Q: {h['question']}")
            st.sidebar.caption(f"A: {h['answer'][:120]}...")
    st.sidebar.download_button(
        "履歴をダウンロード (JSON)",
        data=json.dumps(history, ensure_ascii=False, indent=2),
        file_name="chat_history.json",
        mime="application/json"
    )
else:
    st.sidebar.info("まだ履歴がありません")

# ⭐ 履歴リセット機能
if st.sidebar.button("履歴をリセット"):
    history = []
    save_history(history)
    st.sidebar.success("履歴をリセットしました！")

# --- PDFアップロード ---
uploaded_files = st.file_uploader(
    "PDFをアップロードしてください (複数可)",
    type=["pdf"],
    accept_multiple_files=True
)

all_docs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        text = "".join(p.extract_text() or "" for p in reader.pages)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = [Document(page_content=chunk, metadata={"source": uploaded_file.name})
                for chunk in splitter.split_text(text)]
        all_docs.extend(docs)
    st.success(f"✅ 全PDFの合計チャンク数: {len(all_docs)}")

# --- ベクトルDB作成 ---
if all_docs:
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(all_docs, emb)
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20})

    # --- QA入力 ---
    query = st.text_input("質問を入力してください")
    if query:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([d.page_content for d in docs])

        prompt = f"次の資料に基づいて質問に答えてください。\n\n資料:\n{context}\n\n質問: {query}\n\n答え:"
        answer = llm.predict(prompt)

        st.subheader("📝 回答")
        st.write(answer)

        # --- 履歴に保存 ---
        history.append({"question": query, "answer": answer})
        save_history(history)

        # --- 関連ドキュメント表示 ---
        with st.expander("📖 参照したドキュメント"):
            for d in docs:
                st.markdown(f"**{d.metadata['source']}**: {d.page_content[:200]}...")
