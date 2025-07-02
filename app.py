# app.py - Versión optimizada para poca RAM
import streamlit as st
import os
from dotenv import load_dotenv

# --- Carga de variables de entorno (esto es ligero) ---
load_dotenv()

# --- Funciones de configuración (las importaciones pesadas están DENTRO) ---
@st.cache_resource
def setup_connections_and_embeddings():
    """Carga las librerías pesadas y establece las conexiones una sola vez."""
    # Solo importamos cuando se llama a la función
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_pinecone import Pinecone as LangchainPinecone

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        st.error("Error: PINECONE_API_KEY no encontrada.")
        st.stop()
    
    # Esta es la parte que más memoria consume al inicio
    st.write("Cargando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    st.write("Conectando a Pinecone...")
    vector_store = LangchainPinecone.from_existing_index("asistente-legal-ia", embeddings)
    st.write("Conexión establecida.")
    
    return vector_store

@st.cache_resource
def setup_qa_chain(_vector_store): # Pasamos el vector_store como argumento
    """Carga el LLM y crea la cadena de QA."""
    # Solo importamos cuando se llama a la función
    from langchain_community.llms import HuggingFaceHub
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not HUGGINGFACE_TOKEN:
        st.error("Error: HUGGINGFACEHUB_API_TOKEN no encontrada.")
        st.stop()

    llm = HuggingFaceHub(
        repo_id="google-t5/t5-base", # Usamos t5-base que es más ligero que t5-small
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
        retriever=_vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# --- Interfaz de Usuario (El flujo principal) ---
st.set_page_config(page_title="Rechts-Assistent (Live)", layout="wide")
st.title("🤖 Juristischer Informations-Assistent")
st.caption("Basierend auf dem deutschen Mietrecht | Database: Pinecone")

st.warning("""
**Haftungsausschluss (Disclaimer):** Dies ist ein akademisches Projekt...
""", icon="⚠️")

# El flujo de carga ahora es visible para el usuario
with st.spinner("Inicializando la conexión con la base de conocimiento..."):
    try:
        vector_store = setup_connections_and_embeddings()
        qa_chain = setup_qa_chain(vector_store)
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