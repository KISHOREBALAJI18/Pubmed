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

# ========== Fetch abstracts with XML error handling ==========
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

    try:
        search_response = safe_request(search_url, search_params)
        root = ET.fromstring(search_response.text)
    except ET.ParseError:
        st.error("❌ PubMed search response is not valid XML. Possibly rate-limited or query error.")
        st.code(search_response.text[:1000])
        return [], 0
    except Exception as e:
        st.error(f"❌ Error during search request: {e}")
        return [], 0

    count_text = root.findtext(".//Count")
    if not count_text or not count_text.isdigit():
        st.error("❌ Count value not found in search response.")
        st.code(search_response.text[:1000])
        return [], 0

    count = int(count_text)
    webenv = root.findtext(".//WebEnv")
    query_key = root.findtext(".//QueryKey")

    if not webenv or not query_key:
        st.error("❌ Missing WebEnv or QueryKey in search response.")
        return [], 0

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

        try:
            fetch_response = safe_request(fetch_url, fetch_params)
            fetch_root = ET.fromstring(fetch_response.text)
        except ET.ParseError:
            st.warning(f"⚠️ Skipping batch {start}-{start+batch_size}. Invalid XML.")
            st.code(fetch_response.text[:1000])
            continue
        except Exception as e:
            st.warning(f"⚠️ Fetch batch error {start}-{start+batch_size}: {e}")
            continue

        articles = fetch_root.findall(".//PubmedArticle")

        for article in articles:
            abstract_elem = article_
