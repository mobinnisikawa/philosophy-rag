
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader
import os

st.set_page_config(page_title="哲学RAG", layout="wide")
st.title("📚 哲学RAGシステム (Streamlit版)")

# アップロード
uploaded_file = st.file_uploader("PDFをアップロードしてください", type=["pdf"])
if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = "\n".join([p.extract_text() or "" for p in reader.pages])
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = [Document(page_content=chunk, metadata={"source": uploaded_file.name})
            for chunk in splitter.split_text(text)]

    st.success(f"✅ チャンク数: {len(docs)}")

    # ベクトルDB作成
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(docs, emb)
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k":8, "fetch_k":20})

    # 質問入力
    question = st.text_input("質問を入力してください")
    if question:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate

        SYSTEM_PROMPT = '''あなたは大学院レベルの哲学研究指導者です。
        アップロードされたPDFを根拠に、学術的に正確かつ長文で日本語解説を行ってください。
        '''
        prompt = ChatPromptTemplate.from_template("質問: {input}\n\n# コンテキスト\n{context}\n" + SYSTEM_PROMPT)

        doc_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, doc_chain)

        result = rag_chain.invoke({"input": question})
        st.write(result["answer"])
