from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load extracted text
with open("extracted_text.txt", "r", encoding="utf-8") as f:
    full_text = f.read().strip()  # Ensure no leading/trailing empty spaces

# Initialize Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Max token length per chunk
    chunk_overlap=100  # Overlap to maintain context
)

# Split the text
chunks = text_splitter.split_text(full_text)

# Remove empty chunks
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

# Save chunks for debugging (optional)
with open("text_chunks.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"--- Chunk {i+1} ---\n{chunk}\n\n")

# Print results
print(f"✅ Successfully split text into {len(chunks)} valid chunks!")
