import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.functional as F
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
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
            if len(line.split(maxsplit=1)) == 2 and os.path.isfile(os.path.join(os.path.dirname(trans_file), f"{line.split()[0]}.flac"))
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
    delta = F.compute_deltas(fbank)
    delta_delta = F.compute_deltas(delta)
    features = torch.stack([fbank, delta, delta_delta], dim=-1)
    return features

# -------------------------------
# Evaluation function using WER via torchaudio's CTCDecoder.
# -------------------------------
def evaluate(model, loader, ctc_loss_fn, device, eos_id, sp, decoder):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    # Initialize WER metric
    wer_metric = WordErrorRate()
    with torch.no_grad():
        for batch in loader:
            x, y, x_len, y_len = [b.to(device) for b in batch]
            logits = model(x).transpose(0, 1)    # shape: (T, batch, num_classes)
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

            # Update WER metric with batch results
            wer_metric.update(predicted_transcripts, actual_transcripts)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_wer = wer_metric.compute()  # Get the final WER
    return avg_loss, avg_wer

# -------------------------------
# Training loop for the seed model (Supervised Only)
# -------------------------------
def train_seed(labeled_loader, model, optimizer, ctc_loss_fn, num_epochs, device, val_loader, sp, decoder, eos_id):
    train_losses = []
    val_losses = []
    val_wers = []
    best_val_loss = torch.inf

    # Open a log file to append epoch logs.
    log_file = open("model/weights/seed/seed_training_log.txt", "w")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        pbar = tqdm(labeled_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            x, y, x_len, y_len = [b.to(device) for b in batch]
            logits = model(x).transpose(0, 1)  # (T, batch, num_classes)
            # Convert logits to log-probabilities
            log_probs = torch_F.log_softmax(logits, dim=-1)
            x_len_downsampled = compute_downsampled_time(x_len)
            loss = ctc_loss_fn(log_probs, y, x_len_downsampled, y_len)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        log_msg = f"Epoch {epoch} training loss: {avg_loss:.4f}\n"
        print(log_msg, end="")
        log_file.write(log_msg)

        # Evaluate on validation set using WER.
        val_loss, val_wer = evaluate(model, val_loader, ctc_loss_fn, device, eos_id, sp, decoder)
        val_losses.append(val_loss)
        val_wers.append(val_wer)
        log_msg = f"Epoch {epoch} validation loss: {val_loss:.4f}, WER: {val_wer*100:.2f}%\n"
        print(log_msg, end="")
        log_file.write(log_msg)

        # Save checkpoint if best loss so far
        if val_loss < best_val_loss:
            checkpoint_path = f"model/weights/seed/seed_model_updated.pt"
            torch.save(model.state_dict(), checkpoint_path)
            best_val_loss = val_loss
            log_msg = f"Saved checkpoint to {checkpoint_path}\n"
            print(log_msg, end="")
            log_file.write(log_msg)

        # Save plots for this epoch.
        plt.figure()
        plt.plot(range(1, epoch + 1), train_losses, label="Train Loss")
        plt.plot(range(1, epoch + 1), val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("CTC Loss")
        plt.title("Seed Training Loss")
        plt.legend()
        plt.savefig(f"results/seed_loss.png")
        plt.close()

        plt.figure()
        plt.plot(range(1, epoch + 1), [wer*100 for wer in val_wers], label="Val WER (%)")
        plt.xlabel("Epoch")
        plt.ylabel("WER (%)")
        plt.title("Seed Validation WER")
        plt.legend()
        plt.savefig(f"results/seed_wer.png")
        plt.close()

    log_file.close()
    return train_losses, val_losses, val_wers

# -------------------------------
# Main: set up datasets, model, optimizer, and run seed training.
# -------------------------------
if __name__ == "__main__":
    # Define separate corpus roots for training and validation.
    train_corpus_root = "../../data/LibriSpeech/train-clean-100/"  # "./data/dev/"  # path to training dataset
    val_corpus_root = "../../data/LibriSpeech/dev-clean/"  # "./data/dev/"      # path to validation dataset
    sp_model_path = "spm_model_1k.model"
    max_x_len = 5000
    max_y_len = 500
    batch_size = 50
    num_epochs = 500
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create datasets for training and validation.
    train_dataset = OnlineDataset(train_corpus_root, sp_model_path, max_x_len, max_y_len)
    val_dataset = OnlineDataset(val_corpus_root, sp_model_path, max_x_len, max_y_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = 1000 + 1  # 1000 tokens + blank
    model = ASRModel(max_seq_len=max_x_len).to(device)
    # Define the path to the pre-trained weights (.pt file)
    seed_model_path = "model/weights/seed/seed_model.pt"

    # Check if the seed model file exists
    if os.path.isfile(seed_model_path):
        print(f"Loading seed model weights from {seed_model_path}...")
        model.load_state_dict(torch.load(seed_model_path, map_location=device))
        print("Seed model weights loaded successfully.")
    else:
        print(f"Seed model file not found at {seed_model_path}. Proceeding with random initialization.")
    
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load SentencePiece model (for decoding ground truth) separately.
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)

    # Initialize the CTCDecoder.
    # Create vocabulary list for CTC decoder
    vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

    # Define CTC decoder configuration
    decoder = cuda_ctc_decoder(tokens=vocab, nbest=1, beam_size=20)

    print("CTC Decoder initialized!")
    print("Starting training...")
    # Train the seed model.
    train_losses, val_losses, val_wers = train_seed(train_loader, model, optimizer, ctc_loss_fn,
                                                     num_epochs, device, val_loader, sp, decoder, train_dataset.sp.eos_id())

    print("Seed model training complete.")
