def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100):
    """
    Splits a long string of text into smaller chunks of approximately 'chunk_size'
    characters, with a sliding window 'overlap' so thoughts aren't cut in half.
    """
    chunks = []
    start = 0
    text_length = len(text)
 
    # If the text is shorter than our chunk size, just return it as a single chunk
    if text_length <= chunk_size:
        return [text.strip()] if text.strip() else []
 
    while start < text_length:
        # Define where the chunk ends
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
            
        # Move our starting window forward by chunk_size MINUS the overlap
        start += (chunk_size - overlap)
        
    return chunks