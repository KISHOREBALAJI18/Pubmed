import streamlit as st
import requests
import time
from xml.etree import ElementTree as ET
from Bio import Entrez
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
import os

# ========== Configuration ==========
Entrez.email = "royalkishore37@gmail.com"
NCBI_API_KEY = "2ddfd213108a19322c52587d698ed1230f08"
GOOGLE_API_KEY = "AIzaSyCcg8g0xKZkfrt1JAtnzEdXaw7G4OindqY"
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 500

# ========== Initialize LLM ==========
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    convert_system_message_to_human=True
)

# ========== Safe request with retry ==========
def safe_request(url, params, retries=3, delay=2):
    for i in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response
        except Exception as e:
            if i < retries - 1:
                time.sleep(delay)
            else:
                raise e

# ========== Fetch abstracts ==========
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

    search_response = safe_request(search_url, search_params)
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

        fetch_response = safe_request(fetch_url, fetch_params)
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
st.title("PubMed QA System")
st.markdown("Search PubMed, Embed Results with Google Gemini, and Ask Questions!")

query = st.text_input("🔍 Enter your PubMed search query")

if query:
    with st.spinner("Fetching abstracts and creating vector store..."):
        docs, total_count = fetch_all_pubmed_abstracts(query)
        st.info(f"📄 Retrieved {len(docs)} abstracts out of {total_count} total results.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        split_docs = splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GOOGLE_API_KEY
        )

        vector_store = FAISS.from_documents(split_docs, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            return_source_documents=True
        )

    st.success("✅ Vector DB created. You can now ask questions!")

    user_question = st.text_input("💬 Ask a biomedical question")

    if user_question:
        with st.spinner("Thinking..."):
            response = qa_chain({"query": user_question})
            st.markdown("### 🔍 Answer")
            st.write(response["result"])

            st.markdown("### 📚 Sources")
            for doc in response["source_documents"]:
                st.markdown(f"- [Source]({doc.metadata['source']})")
