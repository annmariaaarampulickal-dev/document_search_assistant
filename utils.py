import nltk
 
# Ensure the sentence tokenizer punctuation data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
 
def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Improved chunking: Uses NLTK sentence boundaries, using your
    original parameter names (chunk_size, overlap) for system compatibility.
    """
    if not text or text.isspace():
        return []
        
    # 1. Break the raw text into complete sentences
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    # SAFEGUARD: Because your API passes character overlap 
    # we convert it to a sentence-count overlap (1 sentence) for this logic.
    sentence_overlap = 1 if overlap > 0 else 0
 
    for sentence in sentences:
        sentence_len = len(sentence)
        
        # 2. Check if adding this sentence exceeds your chunk_size
        if current_length + sentence_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk).strip())
            
            # 3. Step back by 1 sentence for the overlap context
            current_chunk = current_chunk[-sentence_overlap:] if sentence_overlap > 0 else []
            current_length = sum(len(s) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_length += sentence_len
        
    # 4. Grab any leftover sentences at the end
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())
        
    return chunks