
import streamlit as st
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb
from agno.knowledge.embedder.ollama import OllamaEmbedder
from docx import Document
from pathlib import Path
import hashlib
import json
import io
import sqlite3
import uuid

st.set_page_config(
    page_title="Word Document AI Agent",
    page_icon="📄",
    layout="wide"
)

st.title("Word Document AI Agent")

UPLOAD_DIR = Path("./uploads")
REGISTRY_FILE = Path("./processed_documents.json")
CHAT_DB = Path("./chat_history.db")

UPLOAD_DIR.mkdir(exist_ok=True)


def get_connection():
    return sqlite3.connect(CHAT_DB)


def initialize_chat_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id)
                REFERENCES conversations(id)
            )
        """)


def get_all_chats():
    with get_connection() as conn:
        return conn.execute("""
            SELECT id, title
            FROM conversations
            ORDER BY updated_at DESC
        """).fetchall()


def get_chat_messages(chat_id):
    with get_connection() as conn:
        return conn.execute("""
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id
        """, (chat_id,)).fetchall()


def create_chat(title):
    chat_id = str(uuid.uuid4())

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO conversations (id, title)
            VALUES (?, ?)
        """, (chat_id, title[:45]))

    return chat_id


def save_message(chat_id, role, content):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO messages (conversation_id, role, content)
            VALUES (?, ?, ?)
        """, (chat_id, role, content))

        conn.execute("""
            UPDATE conversations
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (chat_id,))


def load_registry():
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def save_registry(registry):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as file:
        json.dump(registry, file, indent=4)


@st.cache_resource
def get_agent():
    knowledge = Knowledge(
        vector_db=ChromaDb(
            collection="word_documents_ollama",
            path="./chroma_db",
            persistent_client=True,
            embedder=OllamaEmbedder(
                id="nomic-embed-text",
                dimensions=768,
                host="http://localhost:11434"
            )
        )
    )

    agent = Agent(
        model=Ollama(
            id="llama3.2:latest",
            host="http://localhost:11434"
        ),
        knowledge=knowledge,
        search_knowledge=True,
        instructions=[
            "Answer questions using the uploaded documents.",
            "Use previous conversation only to understand follow-up questions.",
            "If information is not available in the documents, clearly say so.",
            "Do not make up information."
        ],
        markdown=True
    )

    return agent


initialize_chat_database()

agent = get_agent()

if "selected_chat_id" not in st.session_state:
    st.session_state.selected_chat_id = None


with st.sidebar:
    st.title("💬 Chat History")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.selected_chat_id = None
        st.rerun()

    st.divider()

    chats = get_all_chats()
    selected_chat_id = st.session_state.selected_chat_id

    current_chat = next(
        (chat for chat in chats if chat[0] == selected_chat_id),
        None
    )

    if current_chat:
        st.subheader("Current Chat")

        st.button(
            f"💬 {current_chat[1]}",
            key=f"current_{current_chat[0]}",
            disabled=True,
            use_container_width=True
        )

    st.subheader("Previous Chats")

    previous_chats = [
        chat for chat in chats
        if chat[0] != selected_chat_id
    ]

    if previous_chats:
        for chat_id, title in previous_chats:
            if st.button(
                f"💬 {title}",
                key=f"chat_{chat_id}",
                use_container_width=True
            ):
                st.session_state.selected_chat_id = chat_id
                st.rerun()
    else:
        st.caption("No previous chats yet.")

    st.divider()

    with st.expander("📄 Upload Document"):
        uploaded_file = st.file_uploader(
            "Upload one Word document",
            type=["docx"]
        )

        if uploaded_file:
            if st.button("Process Document"):
                file_bytes = uploaded_file.getvalue()
                file_hash = hashlib.sha256(file_bytes).hexdigest()

                registry = load_registry()

                if file_hash in registry:
                    st.info("Document already processed.")

                else:
                    try:
                        doc = Document(io.BytesIO(file_bytes))

                        text = "\n".join(
                            paragraph.text
                            for paragraph in doc.paragraphs
                            if paragraph.text.strip()
                        )

                        if not text.strip():
                            st.error("No text found in document.")

                        else:
                            with st.spinner("Processing document..."):
                                agent.knowledge.insert(
                                    name=uploaded_file.name,
                                    text_content=text
                                )

                                saved_name = (
                                    f"{file_hash}_"
                                    f"{Path(uploaded_file.name).name}"
                                )

                                saved_path = UPLOAD_DIR / saved_name
                                saved_path.write_bytes(file_bytes)

                                registry[file_hash] = {
                                    "name": uploaded_file.name,
                                    "path": str(saved_path)
                                }

                                save_registry(registry)

                            st.success("Document processed successfully!")

                    except Exception as e:
                        st.error(f"Error: {e}")

    with st.expander("📚 Saved Documents"):
        registry = load_registry()

        if registry:
            for document in registry.values():
                st.write(f"📄 {document['name']}")
        else:
            st.caption("No documents processed yet.")


st.header("AI Chat")

selected_chat_id = st.session_state.selected_chat_id

if selected_chat_id:
    messages = get_chat_messages(selected_chat_id)

    for role, content in messages:
        with st.chat_message(role):
            st.markdown(content)

else:
    st.info("Start a new conversation by asking a question.")

question = st.chat_input("Ask a question about your documents...")

if question:
    chat_id = st.session_state.selected_chat_id

    if not chat_id:
        chat_id = create_chat(question)
        st.session_state.selected_chat_id = chat_id

    previous_messages = get_chat_messages(chat_id)

    history = ""

    for role, content in previous_messages[-10:]:
        history += f"\n{role}: {content}"

    save_message(chat_id, "user", question)

    with st.spinner("Generating answer..."):
        try:
            prompt = f"""
Previous conversation:
{history}

New question:
{question}

Answer the new question using the uploaded document knowledge.
Use the previous conversation only when needed to understand the question.
"""

            response = agent.run(prompt)

            answer = response.content or str(response)

            save_message(chat_id, "assistant", answer)

            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")
