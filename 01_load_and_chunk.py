# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# %%
reader = PdfReader("b.e-cse-batchno-15.pdf")

# %%
text = "\n".join(page.extract_text() for page in reader.pages)

# %%
print(text[:500])

# %%
chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

# %%
for i, chunk in enumerate(chunks[:5], 1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)

# %%
print(f"\nTotal chunks: {len(chunks)}")

# %%
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

# %%
chunks1 = splitter.split_text(text)

# %%
for i, chunk1 in enumerate(chunks1[:5], 1):
    print(f"\n--- Chunk {i} ---")
    print(chunk1)

# Total chunks
print(f"\nTotal chunks: {len(chunks1)}")

# %%


