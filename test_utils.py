from utils import split_text_into_chunks

def test_split_text_into_chunks_exact_length():
    """
    Test that the chunking utility strictly respects the character size limit
    while handling NLTK sentence blocks.
    """
    single_sentence = "This is a dummy sentence used for testing python. "
    sample_text = single_sentence * 24

    chunks = split_text_into_chunks(sample_text, chunk_size=500, overlap=100)

    for chunk in chunks:
        assert len(chunk) <= 500

    assert len(chunks) > 1


def test_chunk_overlap():
    """
    Test that sentence-level overlap works correctly.
    Uses UNIQUE sentences so the overlap assertion cannot pass by coincidence.
    """
    # Each sentence is unique and identifiable by its number
    sentences = [f"This is unique sentence number {i} about topic alpha." for i in range(1, 20)]
    sample_text = " ".join(sentences)

    chunks = split_text_into_chunks(sample_text, chunk_size=300, overlap=100)

    # Must produce at least 2 chunks
    assert len(chunks) >= 2

    # Get the LAST sentence of chunk 1
    # Split by ". " and reconstruct to get proper sentence
    chunk1_sentences = [s.strip() for s in chunks[0].split(".") if s.strip()]
    last_sentence_of_chunk1 = chunk1_sentences[-1] + "."

    # Get the FIRST sentence of chunk 2
    chunk2_sentences = [s.strip() for s in chunks[1].split(".") if s.strip()]
    first_sentence_of_chunk2 = chunk2_sentences[0] + "."

    # The overlap means the last sentence of chunk 1 should
    # ALSO appear as the first sentence of chunk 2
    # This is the actual overlap behavior we're testing
    assert first_sentence_of_chunk2 in chunks[0], (
        f"Overlap failed: '{first_sentence_of_chunk2}' "
        f"should be in chunk 1 but wasn't.\n"
        f"Chunk 1: {chunks[0]}\n"
        f"Chunk 2: {chunks[1]}"
    )


def test_chunk_overlap_disabled():
    """
    Verify that when overlap=0, NO sentence from chunk 2
    appears at the end of chunk 1.
    This test would FAIL with the old test data (repeated sentences)
    but correctly PASSES now because sentences are unique.
    """
    sentences = [f"This is unique sentence number {i} about topic beta." for i in range(1, 20)]
    sample_text = " ".join(sentences)

    chunks = split_text_into_chunks(sample_text, chunk_size=300, overlap=0)

    assert len(chunks) >= 2

    # Get first sentence of chunk 2
    chunk2_sentences = [s.strip() for s in chunks[1].split(".") if s.strip()]
    first_sentence_of_chunk2 = chunk2_sentences[0] + "."

    # With overlap=0, chunk 2's first sentence should NOT
    # appear in chunk 1 at all (no carryover)
    assert first_sentence_of_chunk2 not in chunks[0], (
        f"Expected no overlap but found '{first_sentence_of_chunk2}' in chunk 1"
    )