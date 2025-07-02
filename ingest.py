# app.py - Versión final para producción
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Carga las variables de entorno
load_dotenv()

# --- Configuración de Clientes y Conexiones ---
@st.cache_resource
def setup_connections():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not all([PINECONE_API_KEY, HUGGINGFACE_TOKEN]):
        st.error("Error: Faltan las claves de API en la configuración del entorno.")
        st.stop()
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = LangchainPinecone.from_existing_index("asistente-legal-ia", embeddings)
    
    return vector_store

vector_store = setup_connections()

# --- Configuración de la Cadena de IA ---
@st.cache_resource
def setup_qa_chain():
    llm = HuggingFaceHub(
        repo_id="google-t5/t5-base",
        model_kwargs={"temperature": 0.2, "max_new_tokens": 512}
    )

    prompt_template = """
    Benutze den folgenden Kontext, um die Frage am Ende zu beantworten. Antworte nur auf Deutsch.
    Wenn du die Antwort im Kontext nicht findest, sage: "Ich habe keine Informationen dazu in meiner Wissensdatenbank." Erfinde nichts.

    Kontext:
    {context}

    Frage: {question}
    Hilfreiche Antwort auf Deutsch:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# --- Interfaz de Usuario ---
st.set_page_config(page_title="Rechts-Assistent (Live)", layout="wide")
st.title("🤖 Juristischer Informations-Assistent")
st.caption("Basierend auf dem deutschen Mietrecht | Database: Pinecone")

st.warning("""
**Haftungsausschluss (Disclaimer):** Dies ist ein akademisches Projekt...
""", icon="⚠️")

try:
    qa_chain = setup_qa_chain()
    st.success("Erfolgreich mit der Wissensdatenbank verbunden!")
except Exception as e:
    st.error(f"Verbindung zur Wissensdatenbank fehlgeschlagen: {e}")
    st.stop()

user_question = st.text_input("Stellen Sie hier Ihre Frage zum Mietrecht:", placeholder="z.B. Wie lange ist die Kündigungsfrist?")

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