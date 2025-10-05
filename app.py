import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader
import os
import json

# ===== ページ設定 =====
st.set_page_config(page_title="哲学RAG", layout="wide")
st.title("📚 哲学RAGシステム (Streamlit版)")

# ===== 履歴ファイル =====
HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):  # ここで必ずリストか確認
                return data
            else:
                return []
        except:
            return []
    return []

def save_history(history):
    if not isinstance(history, list):
        history = []
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ===== PDFアップロード =====
uploaded_files = st.file_uploader(
    "PDFをアップロードしてください（複数可）",
    type=["pdf"], accept_multiple_files=True
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

    # ベクトルDB作成
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(all_docs, emb)
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20})

    # ===== 質問入力 =====
    query = st.text_input("❓ 質問を入力してください")
    if query:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([d.page_content for d in docs])

        prompt = f"次の資料を使って質問に答えてください。\n\n資料:\n{context}\n\n質問: {query}\n\n答え:"
        answer = llm.predict(prompt)

        st.subheader("💡 回答")
        st.write(answer)

        # 履歴保存
        history = load_history()
        if not isinstance(history, list):
            history = []
        history.append({"question": query, "answer": answer})
        save_history(history)

        # 参考文献表示
        with st.expander("📑 参照したドキュメント"):
            for d in docs:
                st.markdown(f"- **{d.metadata['source']}**: {d.page_content[:200]}...")

# ===== サイドバーに履歴表示 =====
with st.sidebar:
    st.header("🕑 チャット履歴")

    history = load_history()
    if history:
        st.write("最新5件（下に全件表示 & ダウンロード可）")
        for qa in history[-5:]:
            st.markdown(f"**Q:** {qa['question']}\n\n**A:** {qa['answer'][:100]}...")

        # 全件表示
        if st.checkbox("全履歴を表示する"):
            for i, qa in enumerate(history, 1):
                st.markdown(f"{i}. **Q:** {qa['question']}\n\n　**A:** {qa['answer']}")

        # ダウンロード
        st.download_button(
            label="📥 履歴をダウンロード (JSON)",
            data=json.dumps(history, ensure_ascii=False, indent=2),
            file_name="chat_history.json",
            mime="application/json"
        )
    else:
        st.info("まだ履歴がありません。質問するとここに保存されます。")
