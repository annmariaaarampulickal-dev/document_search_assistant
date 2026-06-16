from utils import split_text_into_chunks
 
def test_split_text_into_chunks_exact_length():
    """
    Test that the chunking utility strictly respects the character size limit
    while handling NLTK sentence blocks.
    """
    # Create a dummy string of actual sentences (Each sentence is 50 characters)
    # 24 sentences * 50 characters = 1200 characters total
    single_sentence = "This is a dummy sentence used for testing python. "
    sample_text = single_sentence * 24 
 
    # Run it through your NLTK utility function
    chunks = split_text_into_chunks(sample_text, chunk_size=500, overlap=100)
 
    # Assertions: Verify that no chunk exceeds 500 characters
    for chunk in chunks:
        assert len(chunk) <= 500
 
    # Verify it actually split the text into multiple pieces
    assert len(chunks) > 1
 
def test_chunk_overlap():
    """
    Test that sentence-level overlap works correctly.
    """
    sentence_a = "This belongs to category A. " * 8  # ~224 characters
    sentence_b = "This belongs to category B. " * 8  # ~224 characters
    sample_text = sentence_a + sentence_b
    
    chunks = split_text_into_chunks(sample_text, chunk_size=300, overlap=100)
 
    # Verify that the second chunk contains at least one sentence from the first chunk
    assert len(chunks) >= 2
    
    # Check if the first sentence of the second chunk exists anywhere in the first chunk
    first_sentence_of_chunk_2 = chunks[1].split(". ")[0] + ". "
    assert first_sentence_of_chunk_2 in chunks[0]
 