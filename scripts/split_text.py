from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text_path):
    with open(text_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(raw_text)

    with open("data/chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.strip() + "\n\n")
    print(f"✅ {len(chunks)} chunks written to data/chunks.txt")
    return chunks

if __name__ == "__main__":
    split_text("data/iso_text.txt")
