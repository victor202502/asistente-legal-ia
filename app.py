# -*- coding: utf-8 -*-
import streamlit as st
import os
from dotenv import load_dotenv

# --- Importaciones de LangChain (para las versiones estables que instalamos) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Para estas versiones, Embeddings y LLM viven en 'langchain_community'
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub

# --- Carga de la API Key de Hugging Face ---
load_dotenv() 

if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    st.error("¡Error! La API key de Hugging Face no se ha encontrado. Asegúrate de tener un archivo .env con HUGGINGFACEHUB_API_TOKEN.")
    st.stop()

# --- Función para configurar la cadena de QA (cacheada para eficiencia) ---
@st.cache_resource
def setup_qa_chain():
    # 1. Cargar el documento PDF
    loader = PyPDFLoader("ley.pdf") # ¡ASEGÚRATE QUE ESTE ES EL NOMBRE DE TU PDF!
    docs = loader.load()

    # 2. Dividir el texto en chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)

    # 3. Crear embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Crear la base de datos vectorial
    vectorstore = Chroma.from_documents(chunks, embeddings)

    # 5. Configurar el LLM (el modelo de lenguaje)
    llm = HuggingFaceHub(
         repo_id="google-t5/t5-small", # ¡NUESTRO MODELO FIABLE!
    model_kwargs={"temperature": 0.2, "max_new_tokens": 512}
    )

    # 6. Crear la cadena de QA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# --- Interfaz de Usuario con Streamlit ---
st.set_page_config(page_title="Rechts-Assistent (Beta)", layout="wide")
st.title("🤖 Juristischer Informations-Assistent (Beta)")
st.caption("Basierend auf dem deutschen Mietrecht (BGB §§ 535-580a)")

st.warning("""
**Haftungsausschluss (Disclaimer):** Dies ist ein akademisches Projekt und bietet keine Rechtsberatung. 
Die generierten Informationen können ungenau oder veraltet sein. Konsultieren Sie immer einen qualifizierten Anwalt.
""", icon="⚠️")

try:
    qa_chain = setup_qa_chain()
    st.success("Die Wissensdatenbank wurde erfolgreich geladen!")
except Exception as e:
    st.error(f"Ein Fehler ist beim Laden der Wissensdatenbank aufgetreten: {e}")
    st.stop()

user_question = st.text_input("Stellen Sie hier Ihre Frage zum Mietrecht:", placeholder="z.B. Wie lange ist die Kündigungsfrist für meine Wohnung?")

if user_question:
    with st.spinner("Suche in der Gesetzesdatenbank und generiere eine Antwort..."):
        try:
            # Con estas versiones, es más seguro llamar a la cadena así:
            result = qa_chain({"query": user_question})
            
            st.subheader("Antwort:")
            st.write(result["result"])

            with st.expander("Quellen anzeigen (verwendete Textabschnitte)"):
                for doc in result["source_documents"]:
                    st.info(f"Quelle (Seite {doc.metadata.get('page', 'N/A')}):")
                    st.text(doc.page_content)

        except Exception as e:
            st.error(f"Entschuldigung, bei der Beantwortung Ihrer Frage ist ein Fehler aufgetreten: {e}")