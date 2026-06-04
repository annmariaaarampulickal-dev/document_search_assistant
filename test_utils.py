from utils import split_text_into_chunks
 
def test_split_text_into_chunks_exact_length():
    """
    Test that the chunking utility strictly respects the character size limit.
    """
    # Create a dummy string that is 1200 characters long
    sample_text = "abcdefghij" * 120 
    
    # Run it through your utility function with your exact project rules
    chunks = split_text_into_chunks(sample_text, chunk_size=500, overlap=100)
    
    # Assertions: Verify that no chunk exceeds 500 characters
    for chunk in chunks:
        assert len(chunk) <= 500
        
    # Verify it actually split the text into multiple pieces
    assert len(chunks) > 1
 
def test_chunk_overlap():
    """
    Test that the 100-character sliding window overlap works correctly.
    """
    sample_text = "A" * 400 + "B" * 400
    chunks = split_text_into_chunks(sample_text, chunk_size=500, overlap=100)
    
    # If overlap works, the second chunk must contain trailing elements of the first chunk
    assert len(chunks) >= 2
    # The start of the second chunk should overlap with the end of the first chunk
    assert chunks[1][:10] == chunks[0][-100:-90]
 