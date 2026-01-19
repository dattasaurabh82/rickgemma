import fitz  # This is PyMuPDF
import json
import re

# CONFIGURATION
# ---------------------
PDF_PATH = "book.pdf"
OUTPUT_DIR = "data"
CHUNK_SIZE = 200
AUTHOR_STYLE_PROMPT = "You are the author of this book. Share your thoughts." 
# ---------------------

def clean_text(text):
    # Basic cleaning
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_dataset():
    print(f"📖 Reading {PDF_PATH} using PyMuPDF...")
    
    try:
        doc = fitz.open(PDF_PATH)
    except Exception as e:
        print(f"❌ Error opening PDF: {e}")
        return

    full_text = ""
    
    # Extract text from every page
    for page in doc:
        full_text += page.get_text() + " "
    
    words = clean_text(full_text).split()
    print(f"📝 Extracted {len(words)} words. Chunking...")

    if len(words) == 0:
        print("❌ Error: No text found! Is this a scanned image PDF? If so, we need OCR.")
        return

    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= CHUNK_SIZE:
            chunks.append(" ".join(current_chunk))
            current_chunk = []  # Reset
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"💾 Saving {len(chunks)} chunks to {OUTPUT_DIR}/train.jsonl...")
    
    with open(f"{OUTPUT_DIR}/train.jsonl", "w") as f:
        for chunk in chunks:
            entry = {
                "text": f"<start_of_turn>user\n{AUTHOR_STYLE_PROMPT}<end_of_turn>\n<start_of_turn>model\n{chunk}<end_of_turn>"
            }
            f.write(json.dumps(entry) + "\n")
            
    with open(f"{OUTPUT_DIR}/valid.jsonl", "w") as f:
        # Take a random sample or just the first few for validation
        for i, chunk in enumerate(chunks[:10]):
            entry = {
                "text": f"<start_of_turn>user\n{AUTHOR_STYLE_PROMPT}<end_of_turn>\n<start_of_turn>model\n{chunk}<end_of_turn>"
            }
            f.write(json.dumps(entry) + "\n")

    print("✅ Done! Ready for training.")

if __name__ == "__main__":
    create_dataset()


