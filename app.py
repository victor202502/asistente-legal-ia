# -*- coding: utf-8 -*-
import streamlit as st
import os
from dotenv import load_dotenv

# Nuevas importaciones para Pinecone
from pinecone import Pinecone as PineconeClient
from langchain_community.vectorstores import Pinecone as LangchainPinecone

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Carga las variables de entorno
load_dotenv()

# --- Configuración de Clientes (Pinecone y Hugging Face) ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not all([PINECONE_API_KEY, HUGGINGFACE_TOKEN]):
    st.error("¡Error! Faltan las claves de API. Revisa PINECONE_API_KEY y HUGGINGFACEHUB_API_TOKEN.")
    st.stop()
    
# Inicializa el cliente de Pinecone (versión Serverless)
pinecone = PineconeClient(api_key=PINECONE_API_KEY)

# Nombre de nuestro índice en Pinecone
INDEX_NAME = "asistente-legal-ia"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Función principal de la App ---
@st.cache_resource
def setup_qa_chain():
    # Conectamos con nuestro índice existente en Pinecone
    vector_store = LangchainPinecone.from_existing_index(INDEX_NAME, embeddings)
    
    llm = HuggingFaceHub(
        repo_id="google-t5/t5-base",
        model_kwargs={"temperature": 0.2, "max_new_tokens": 512}
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# --- Interfaz de Usuario ---
st.set_page_config(page_title="Rechts-Assistent (Live)", layout="wide")
st.title("🤖 Juristischer Informations-Assistent (Live Version)")
st.caption("Basierend auf dem deutschen Mietrecht (BGB §§ 535-580a) | Database: Pinecone Serverless")

st.warning("""
**Haftungsausschluss (Disclaimer):** Dies ist ein akademisches Projekt und bietet keine Rechtsberatung. 
Die generierten Informationen können ungenau oder veraltet sein. Konsultieren Sie immer einen qualifizierten Anwalt.
""", icon="⚠️")

# Botón para cargar el documento (SOLO SE HACE UNA VEZ)
if st.button("Dokumentenbasis erstmalig aufbauen (nur einmal klicken)"):
    with st.spinner("Lade und verarbeite PDF... Dies kann einige Minuten dauern. Bitte warten."):
        try:
            loader = PyPDFLoader("ley.pdf") # ASEGÚRATE QUE EL NOMBRE ES CORRECTO
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = text_splitter.split_documents(documents)
            
            # Usamos 'from_documents' para crear y poblar el índice por primera vez
            LangchainPinecone.from_documents(docs, embeddings, index_name=INDEX_NAME)
            st.success("Die Wissensdatenbank wurde erfolgreich in Pinecone erstellt!")
        except Exception as e:
            st.error(f"Fehler beim Aufbau der Datenbank: {e}")

try:
    qa_chain = setup_qa_chain()
    st.success("Erfolgreich mit der Pinecone-Wissensdatenbank verbunden!")
except Exception as e:
    st.error(f"Verbindung zur Pinecone-Datenbank fehlgeschlagen: {e}. Haben Sie die Dokumentenbasis bereits aufgebaut?")
    st.stop()


user_question = st.text_input("Stellen Sie hier Ihre Frage zum Mietrecht:", placeholder="z.B. Wie lange ist die Kündigungsfrist für meine Wohnung?")

if user_question:
    with st.spinner("Suche in der Cloud-Datenbank und generiere eine Antwort..."):
        try:
            result = qa_chain({"query": user_question})
            st.subheader("Antwort:")
            st.write(result["result"])
            with st.expander("Quellen anzeigen"):
                for doc in result["source_documents"]:
                    st.info(f"Quelle (Seite {doc.metadata.get('page', 'N/A')}):")
                    st.text(doc.page_content)
        except Exception as e:
            st.error(f"Ein Fehler ist aufgetreten: {e}")