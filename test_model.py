import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.functional as F
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as torch_F
from torcheval.metrics import WordErrorRate
from model.model import ASRModel
from torchaudio.models.decoder import cuda_ctc_decoder


def compute_downsampled_time(T):
    '''
    Helper function to compute expected temporal length after downsampling by 4.
    '''
    # Conv1: kernel_size=3, stride=2, padding=0
    H_1 = torch.floor((T - 3) / 2) + 1
    # Conv2: kernel_size=3, stride=2, padding=0
    H_2 = torch.floor((H_1 - 3) / 2) + 1
    return H_2.to(torch.int32)


# -------------------------------
# Dataset: OnlineDataset reads from a given corpus_root.
# -------------------------------
class OnlineDataset(Dataset):
    def __init__(self, corpus_root, sp_model_path, max_x_len=300, max_y_len=500, sample_rate=16000):
        self.corpus_root = corpus_root
        self.max_x_len = max_x_len
        self.max_y_len = max_y_len
        self.sample_rate = sample_rate

        # Load SentencePiece model
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(sp_model_path)
        self.eos_id = self.sp.eos_id()  # End-of-sequence token

        # Collect all samples (audio paths and transcripts)
        import glob
        trans_pattern = os.path.join(corpus_root, "**", "*.trans.txt")
        trans_files = glob.glob(trans_pattern, recursive=True)
        self.samples = [
            {
                "audio_path": os.path.join(os.path.dirname(trans_file), f"{line.split()[0]}.flac"),
                "transcript": line.split(maxsplit=1)[1]
            }
            for trans_file in trans_files
            for line in open(trans_file, "r", encoding="utf-8").read().strip().split("\n")
            if len(line.split(maxsplit=1)) == 2 and os.path.isfile(
                os.path.join(os.path.dirname(trans_file), f"{line.split()[0]}.flac"))
        ]
        print(f"Total samples found in {corpus_root}: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample["audio_path"]
        transcript = sample["transcript"]

        # Load audio
        wave, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            wave = resampler(wave)

        # Extract features with deltas
        features = extract_features_with_deltas(wave, self.sample_rate)
        # Pad/truncate features to max_x_len and record true length.
        x = torch.zeros((self.max_x_len, 80, 3))
        x_len = min(features.size(0), self.max_x_len)
        x[:x_len] = features[:x_len]

        # Tokenize transcript and pad to max_y_len.
        token_ids = torch.tensor(self.sp.EncodeAsIds(transcript), dtype=torch.long)
        y = torch.full((self.max_y_len,), fill_value=self.eos_id, dtype=torch.long)
        y_len = min(token_ids.size(0), self.max_y_len)
        y[:y_len] = token_ids[:y_len]

        return x, y, x_len, y_len


def extract_features_with_deltas(wave, sample_rate):
    """
    Extracts 80-dim filterbank features, delta, and delta-delta.
    Returns a tensor of shape (num_frames, 80, 3).
    """
    if wave.ndim > 1:
        wave = wave[0]
    fbank = torchaudio.compliance.kaldi.fbank(
        wave.unsqueeze(0),
        num_mel_bins=80,
        sample_frequency=sample_rate
    )
    # print(fbank.shape)
    delta = F.compute_deltas(fbank)
    delta_delta = F.compute_deltas(delta)
    features = torch.stack([fbank, delta, delta_delta], dim=-1)
    return features


def evaluate(model, loader, ctc_loss_fn, device, eos_id, sp, decoder):
    model.eval()
    total_loss = 0.0
    # total_wer = 0.0
    num_batches = 0
    # Initialize WER metric
    wer_metric = WordErrorRate()
    with torch.no_grad():
        for batch in loader:
            x, y, x_len, y_len = [b.to(device) for b in batch]
            logits = model(x).transpose(0, 1)  # shape: (T, batch, num_classes)
            # Convert logits to log-probabilities
            log_probs = torch_F.log_softmax(logits, dim=-1)
            x_len_downsampled = compute_downsampled_time(x_len)
            loss = ctc_loss_fn(log_probs, y, x_len_downsampled, y_len)
            total_loss += loss.item()
            num_batches += 1

            log_probs = log_probs.transpose(0, 1).contiguous()

            # Perform beam search decoding for the entire batch
            predicted_tokens = decoder(log_probs.to(device), x_len_downsampled.to(device))

            # Decode actual and predicted transcripts for all samples in the batch directly
            actual_transcripts = sp.decode([y[i].tolist() for i in range(len(y))])
            predicted_transcripts = sp.decode([predicted_tokens[i][0].tokens for i in range(len(predicted_tokens))])
            scores = [predicted_tokens[i][0].score for i in range(len(predicted_tokens))]

            # Update WER metric with batch results
            wer_metric.update(predicted_transcripts, actual_transcripts)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_wer = wer_metric.compute()  # Get the final WER
    return avg_loss, avg_wer


# -------------------------------
# Main: set up datasets, model, optimizer, and run test.
# -------------------------------
if __name__ == "__main__":
    # Define corpus for testing.
    test_corpus_root = "../../data/LibriSpeech/test-clean/"  # Update with your test set path.
    sp_model_path = "spm_model_1k.model"
    max_x_len = 5000
    max_y_len = 500
    batch_size = 50
    num_epochs = 500
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset for testing.
    test_dataset = OnlineDataset(test_corpus_root, sp_model_path, max_x_len, max_y_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = 1000 + 1  # 1000 tokens + blank
    model = ASRModel(max_seq_len=max_x_len).to(
        device)
    # Define the path to the pre-trained weights (.pt file)
    model_path = "model/weights/mpl/mpl_model.pt"

    # Check if the model file exists
    if os.path.isfile(model_path):
        print(f"Loading seed model weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully.")
    else:
        print(f"Model file not found at {model_path}. Proceeding with random initialization.")

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    # Load SentencePiece model (for decoding ground truth) separately.
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)

    # Initialize the CTCDecoder.
    # Create vocabulary list for CTC decoder
    vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

    # Define CTC decoder configuration
    decoder = cuda_ctc_decoder(tokens=vocab, nbest=1, beam_size=20)

    print("CTC Decoder initialized!")

    test_loss, test_wer = evaluate(model, test_loader, ctc_loss_fn, device, test_dataset.sp.eos_id, sp, decoder)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test WER: {test_wer * 100:.2f}%")
