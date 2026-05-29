# Standard LangChain imports for building a RAG (Retrieval-Augmented Generation) pipeline
from langchain_community.document_loaders import PyPDFLoader       # Loads and parses PDF files page by page
from langchain_text_splitters import RecursiveCharacterTextSplitter # Splits long text into smaller overlapping chunks
from langchain_openai import OpenAIEmbeddings, ChatOpenAI           # OpenAI embedding model and chat LLM
from langchain_chroma import Chroma                                  # ChromaDB vector store for similarity search
from langchain_core.prompts import ChatPromptTemplate               # Builds structured prompt templates
from langchain_core.output_parsers import StrOutputParser           # Converts LLM response object to plain string
from langchain_core.runnables import RunnablePassthrough            # Passes input unchanged through a chain step
from dotenv import load_dotenv
load_dotenv()   # Reads OPENAI_API_KEY (and other vars) from a local .env file into os.environ

# ── 1. LOAD & CHUNK ──────────────────────────────────────────────────────────

# Load every page of the PDF as a separate LangChain Document object
# Returns: list[Document], where each Document has .page_content (str) and .metadata (dict with "page" key)
reader = PyPDFLoader("b.e-cse-batchno-15.pdf").load()

# Configure the text splitter:
#   separators  – tries to split on "\n\n" first, then "\n", then " ", then individual characters
#   chunk_size  – max characters per chunk (300 keeps chunks small for precise retrieval)
#   chunk_overlap – characters shared between consecutive chunks (0 = no overlap here)
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=300,
    chunk_overlap=0
)

# Split the loaded pages into chunks
# Returns: list[Document] — same structure as `reader` but each .page_content is ≤300 chars
chunks = text_splitter.split_documents(reader)

# Restrict to the first 10 chunks to limit embedding API calls during experimentation
chunks_10 = chunks[:10]

# ── 2. VECTOR STORE ──────────────────────────────────────────────────────────

# Embed each chunk with text-embedding-3-large and store vectors in a local ChromaDB collection.
# - chunks_10              : the list of Document objects to embed and index
# - OpenAIEmbeddings(...)  : converts each chunk's text to a 3072-dim float vector via OpenAI API
# - collection_name        : logical name for this set of vectors inside ChromaDB
# - persist_directory      : folder path where ChromaDB writes its SQLite + parquet files on disk
# Returns: Chroma — a vector store object ready for similarity search
vector_store = Chroma.from_documents(
    chunks_10,
    OpenAIEmbeddings(model="text-embedding-3-large"),
    collection_name="pdf_chunks",
    persist_directory="./chroma_db"
)

# Confirm how many vectors were actually stored (should equal len(chunks_10))
print(f"Chunks in store: {vector_store._collection.count()}\n")

# ── 3. BUILD RAG CHAIN ───────────────────────────────────────────────────────

# Wrap the vector store as a LangChain retriever.
# search_kwargs={"k": 3} means: for every query, return the 3 most similar chunks by cosine distance.
# Returns: VectorStoreRetriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Define the prompt sent to the LLM.
# {context} will be filled with the retrieved chunk text; {question} with the user's query.
# "Answer based ONLY on context" instructs the model not to hallucinate beyond what was retrieved.
prompt = ChatPromptTemplate.from_template(
    "Answer based ONLY on context: {context}. Question: {question}"
)

# Instantiate GPT-4o-mini as the answer-generation model.
# temperature=0 makes outputs deterministic (no randomness), which is ideal for factual Q&A.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def format_docs(docs):
    """
    Concatenate a list of retrieved Documents into a single context string for the prompt.

    Each document's text is separated by a horizontal rule so the LLM can clearly
    distinguish chunk boundaries when reading the context.

    Parameters
    ----------
    docs : list[Document]
        Retrieved Document objects, each with a .page_content string attribute.

    Returns
    -------
    str
        All page_content values joined by "\\n\\n---\\n\\n".
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# Assemble the full RAG chain using LangChain's pipe (|) operator:
#   Step 1 – Build a dict:
#       "context"  : run the retriever on the question, then format the docs into one string
#       "question" : pass the original question through unchanged (RunnablePassthrough)
#   Step 2 – Fill the prompt template with the dict values
#   Step 3 – Send the filled prompt to GPT-4o-mini and get a response object
#   Step 4 – Parse the response object into a plain Python string
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── 4. TEST 5 QUESTIONS ──────────────────────────────────────────────────────

# Five questions designed to test different retrieval/generation scenarios:
questions = [
    "What is this document about?",                      # Q1 – broad/safe: should always return something
    "What are the names of the students listed?",        # Q2 – factual lookup: tests entity extraction
    "What is the batch number mentioned?",               # Q3 – specific detail: tests precise recall
    "What programming languages are covered?",           # Q4 – may be in later chunks (⚠ failure candidate)
    "What is the customer churn rate?",                  # Q5 – out-of-domain bait: tests hallucination resistance
]

SEPARATOR = "=" * 70  # Visual divider printed between each question block

for i, question in enumerate(questions, 1):
    # Print question header
    print(f"\n{SEPARATOR}")
    print(f"Q{i}: {question}")
    print(SEPARATOR)

    # Retrieve the top-3 most relevant chunks for this question (does NOT call the LLM yet)
    # Returns: list[Document] of length ≤ k (3)
    retrieved_docs = retriever.invoke(question)

    print("\n📄 Top 3 Retrieved Chunks (first 100 chars):")
    for rank, doc in enumerate(retrieved_docs, 1):
        snippet = doc.page_content[:100].replace("\n", " ")  # Truncate + flatten newlines for display
        page    = doc.metadata.get("page", "?")              # PDF page number the chunk came from
        print(f"  [{rank}] (page {page}) {snippet!r}")

    # Run the full RAG chain: retrieve → format → prompt → LLM → parse
    # Returns: str — the model's answer grounded in the retrieved context
    answer = rag_chain.invoke(question)
    print(f"\n🤖 Answer: {answer}")