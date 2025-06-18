import streamlit as st
import requests
from xml.etree import ElementTree as ET
from Bio import Entrez
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from chromadb.config import Settings
from chromadb import PersistentClient
from langchain.vectorstores import Chroma
import os

# ========== Configuration ==========
Entrez.email = "royalkishore37@gmail.com"
NCBI_API_KEY = "2ddfd213108a19322c52587d698ed1230f08"
GOOGLE_API_KEY = "AIzaSyCcg8g0xKZkfrt1JAtnzEdXaw7G4OindqY"
CHROMA_DB_DIR = "./chroma_db"
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 500

# ========== Initialize Gemini Model ==========
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    convert_system_message_to_human=True
)

# ========== Fetch PubMed Abstracts ==========
@st.cache_data(show_spinner=True)
def fetch_all_pubmed_abstracts(query):
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    search_params = {
        "db": "pubmed",
        "term": query,
        "usehistory": "y",
        "retmode": "xml",
        "api_key": NCBI_API_KEY
    }
    search_response = requests.get(search_url, params=search_params)
    root = ET.fromstring(search_response.text)
    count = int(root.findtext(".//Count"))
    webenv = root.findtext(".//WebEnv")
    query_key = root.findtext(".//QueryKey")

    documents = []
    batch_size = 200
    for start in range(0, count, batch_size):
        fetch_params = {
            "db": "pubmed",
            "retmode": "xml",
            "retstart": start,
            "retmax": batch_size,
            "query_key": query_key,
            "WebEnv": webenv,
            "api_key": NCBI_API_KEY
        }
        fetch_response = requests.get(fetch_url, params=fetch_params)
        fetch_root = ET.fromstring(fetch_response.text)
        articles = fetch_root.findall(".//PubmedArticle")

        for article in articles:
            abstract_elem = article.find(".//Abstract/AbstractText")
            pmid_elem = article.find(".//PMID")
            if abstract_elem is not None and abstract_elem.text and pmid_elem is not None:
                abstract = abstract_elem.text.strip()
                pmid = pmid_elem.text.strip()
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                metadata = {"source": url}
                documents.append(Document(page_content=abstract, metadata=metadata))
    return documents, count

# ========== Streamlit UI ==========
st.title("🧠 PubMed QA with Google Gemini")
st.markdown("Search PubMed, Embed Results with Gemini Embeddings, and Ask Questions!")

query = st.text_input("🔍 Enter your PubMed search query")

if query:
    with st.spinner("🔄 Fetching abstracts and preparing vector DB..."):
        docs, total_count = fetch_all_pubmed_abstracts(query)
        st.info(f"📄 Retrieved {len(docs)} abstracts out of {total_count} total PubMed results.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        split_docs = splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GOOGLE_API_KEY
        )

        # ✅ Use DuckDB for compatibility with Streamlit Cloud
        chroma_settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=CHROMA_DB_DIR
        )
        client = PersistentClient(path=CHROMA_DB_DIR, settings=chroma_settings)
        vector_store = Chroma(
            client=client,
            collection_name="pubmed",
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_DIR,
            client_settings=chroma_settings
        )
        vector_store.add_documents(split_docs)
        vector_store.persist()

        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            return_source_documents=True
        )

    st.success("✅ Vector DB created! Ask any biomedical question below.")

    user_question = st.text_input("💬 Ask your biomedical question")

    if user_question:
        with st.spinner("💡 Generating answer..."):
            response = qa_chain({"query": user_question})
            st.markdown("### 🧠 Answer")
            st.write(response["result"])

            st.markdown("### 📚 Sources")
            for doc in response["source_documents"]:
                st.markdown(f"- [PubMed Link]({doc.metadata['source']})")
