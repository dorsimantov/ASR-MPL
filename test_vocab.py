import sentencepiece as spm


def load_sentencepiece_model(model_prefix):
    """
    Loads the SentencePiece model and returns the processor object.
    """
    sp = spm.SentencePieceProcessor()
    model_path = f"{model_prefix}.model"
    vocab_path = f"{model_prefix}.vocab"

    sp.load(model_path)
    print(f"Loaded SentencePiece model: {model_path}")
    print(f"Loaded vocabulary: {vocab_path}")

    return sp


def find_oov_tokens(sp, sentences):
    print("\nOut-of-Vocabulary (OOV) Tokens:")
    for sentence in sentences:
        tokens = sp.encode(sentence, out_type=str)
        oov_tokens = [token for token in tokens if sp.piece_to_id(token) == 0]
        if oov_tokens:
            print(f"Sentence: {sentence}")
            print(f"OOV Tokens: {oov_tokens}\n")


def test_tokenization(sp, sentences):
    """
    Tests tokenization and detokenization using the SentencePiece model.
    """
    print("\nTesting tokenization and detokenization:")
    for sentence in sentences:
        # Tokenize the sentence
        tokens = sp.encode(sentence, out_type=str)
        ids = sp.encode(sentence, out_type=int)

        # Detokenize back to text
        detokenized = sp.decode(ids)

        # Print results
        print("\nOriginal Sentence: ", sentence)
        print("Token IDs: ", ids)
        print("Tokens: ", tokens)
        print("Detokenized: ", detokenized)

        # Check if detokenization is consistent
        if detokenized != sentence:
            print("Warning: Detokenization mismatch!")


def print_vocabulary(sp, num_words=20):
    """
    Prints a sample of the vocabulary.
    """
    print("\nSample of Vocabulary:")
    for i in range(min(num_words, sp.get_piece_size())):
        print(f"{i}: {sp.id_to_piece(i)}")


if __name__ == "__main__":
    model_prefix = "spm_model_1k"  # Ensure this matches your model prefix

    # Load trained SentencePiece model
    sp = load_sentencepiece_model(model_prefix)

    # Print a sample of the vocabulary
    print_vocabulary(sp, num_words=20)

    # Define some sample sentences to test the model
    sample_sentences = [
        "This is a test sentence".upper(),
        "How well does the model handle various sentences".upper(),
        "Let's see how it performs on longer texts with more complexity".upper(),
    ]

    find_oov_tokens(sp, sample_sentences)

    # Test tokenization and detokenization
    test_tokenization(sp, sample_sentences)
