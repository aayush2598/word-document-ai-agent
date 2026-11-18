
import streamlit as st
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb
from docx import Document
import tempfile
from agno.knowledge.embedder.ollama import OllamaEmbedder

st.set_page_config(
    page_title="Word Document AI Agent",
    page_icon="📄",
    layout="wide"
)

st.title("Word Document AI Agent")

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
            "Answer questions only using the uploaded document.",
            "If the answer is not available in the document, say that clearly.",
            "Do not make up information."
        ],
        markdown=True
    )

   

    return agent

agent = get_agent()

uploaded_file = st.file_uploader(
    "Upload Word Document",
    type=["docx"]
)

if uploaded_file:
    if st.button("Process Document"):
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".docx"
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            file_path = temp_file.name

        try:
            doc = Document(file_path)

            text = "\n".join(
                paragraph.text
                for paragraph in doc.paragraphs
                if paragraph.text.strip()
            )

            if not text.strip():
                st.error("No text found in document.")
            else:
                knowledge = agent.knowledge
                knowledge.insert(
                    name=uploaded_file.name,
                    text_content=text
                )
                st.session_state["document_processed"] = True
                st.success("Document processed successfully!")

        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state.get("document_processed", False):
    question = st.text_input("Ask a question about your document")

    if st.button("Ask"):
        if question.strip():
            with st.spinner("Generating answer..."):
                response = agent.run(question)
                st.markdown("### Answer")
                st.markdown(response.content)
