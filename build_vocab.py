import os
import glob
import pandas as pd
import sentencepiece as spm


def gather_transcripts(corpus_root, dataset_name):
    """
    Recursively traverses the corpus root to find all .trans.txt files (from LibriSpeech train-clean-100)
    and extracts the transcript texts.

    Each line in a .trans.txt file has the format:
        <utterance_id> <transcript>
    This function extracts the transcript (everything after the first space).

    Returns:
        A list of transcript strings.
    """
    transcripts = []
    # Recursively search for all .trans.txt files in the train-clean-100 folder.
    pattern = os.path.join(corpus_root, dataset_name, "**", "*.trans.txt")
    print(f"Looking for .trans.txt files in: {pattern}")
    for trans_file in glob.glob(pattern, recursive=True):
        print(f"Found file: {trans_file}")
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    _, transcript = parts
                    transcripts.append(transcript)
    return transcripts


def build_sentencepiece_vocab(corpus_root, vocab_size, output_prefix):
    """
    Builds a SentencePiece vocabulary from all transcripts found under the given corpus root.
    Saves the model with the specified output prefix.
    """
    transcripts = gather_transcripts(corpus_root, dataset_name)
    temp_text_file = "transcripts_temp.txt"
    with open(temp_text_file, "w", encoding="utf-8") as f:
        for transcript in transcripts:
            f.write(transcript.strip() + "\n")

    # Train SentencePiece model with the given vocabulary size.
    spm.SentencePieceTrainer.train(
        input=temp_text_file,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type='bpe',
        bos_id=-1,
        unk_id=0,
        eos_id=1,
        add_dummy_prefix=False,
    )

    os.remove(temp_text_file)
    print(f"SentencePiece model and vocab saved with prefix: {output_prefix}")


if __name__ == "__main__":
    # Set the corpus root (the folder that contains dev-clean)
    dataset_name = "dev-clean"
    corpus_root = "../../data/LibriSpeech/"  # update this path accordingly
    vocab_size = 1000
    output_prefix = "spm_model_1k"
    build_sentencepiece_vocab(corpus_root, vocab_size, output_prefix)
