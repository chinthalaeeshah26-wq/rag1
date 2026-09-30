import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Mini RAG Q&A",
    page_icon="📚"
)

st.title("📚 Mini RAG Q&A")
st.write(
    "Paste a document, store it in ChromaDB, "
    "and ask questions about it."
)

# --------------------------------------------------
# Gemini API
# --------------------------------------------------

try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    gemini_client = genai.Client(api_key=gemini_api_key)
except Exception:
    gemini_client = None

# --------------------------------------------------
# Load Embedding Model
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()

# --------------------------------------------------
# ChromaDB
# --------------------------------------------------

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="documents"
)

# --------------------------------------------------
# Add Document
# --------------------------------------------------

document = st.text_area(
    "📄 Paste your document here",
    height=250,
    placeholder="Paste your notes, article, syllabus, etc..."
)

if st.button("Add Document ➕"):

    if not document.strip():
        st.warning("Please enter some text.")

    else:

        # Split document into chunks
        chunks = [
            document[i:i + 500]
            for i in range(0, len(document), 500)
        ]

        # Create embeddings
        embeddings = embedding_model.encode(chunks)

        # Add chunks to ChromaDB
        collection.add(
            ids=[
                f"chunk_{i}"
                for i in range(len(chunks))
            ],
            documents=chunks,
            embeddings=embeddings.tolist()
        )

        st.success(
            f"Added {len(chunks)} chunk(s) to ChromaDB."
        )

# --------------------------------------------------
# Ask Question
# --------------------------------------------------

question = st.text_input(
    "❓ Ask a question about your document"
)

if st.button("🔍 Ask AI"):

    if not question.strip():
        st.warning("Please enter a question.")

    elif collection.count() == 0:
        st.warning("Please add a document first.")

    elif gemini_client is None:
        st.error(
            "Gemini API key is not configured. "
            "Add GEMINI_API_KEY to Streamlit Secrets."
        )

    else:

        # --------------------------------------------------
        # Create embedding for question
        # --------------------------------------------------

        question_embedding = embedding_model.encode(
            [question]
        )[0]

        # --------------------------------------------------
        # Retrieve relevant chunks
        # --------------------------------------------------

        results = collection.query(
            query_embeddings=[
                question_embedding.tolist()
            ],
            n_results=min(3, collection.count())
        )

        retrieved_chunks = results["documents"][0]

        context = "\n\n".join(
            retrieved_chunks
        )

        # --------------------------------------------------
        # Create RAG Prompt
        # --------------------------------------------------

        prompt = f"""
You are a helpful AI assistant.

Answer the question ONLY using the context provided below.

If the answer is not present in the context,
say exactly:

"I don't know based on the provided document."

Do not use outside knowledge.

Context:
{context}

Question:
{question}
"""

        # --------------------------------------------------
        # Ask Gemini
        # --------------------------------------------------

        try:

            response = gemini_client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            st.subheader("🤖 Answer")

            st.write(response.text)

            # --------------------------------------------------
            # Show Retrieved Context
            # --------------------------------------------------

            with st.expander("🔍 Retrieved Context"):

                for i, chunk in enumerate(
                    retrieved_chunks
                ):

                    st.write(
                        f"**Chunk {i + 1}:**"
                    )

                    st.write(chunk)

        except Exception as e:

            st.error(
                "Could not connect to Gemini API."
            )

            st.code(str(e))


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Python + Sentence Transformers + ChromaDB + Gemini + Streamlit"
)