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


##########################################
# Helper: compute expected time after downsampling by 4.
##########################################
def compute_downsampled_time(T):
    # First 2D convolution: kernel_size=3, stride=2, no padding
    H_1 = torch.floor((T - 3) / 2) + 1
    # Second 2D convolution: kernel_size=3, stride=2, no padding
    H_2 = torch.floor((H_1 - 3) / 2) + 1
    return H_2.to(torch.int32)


##########################################
# Dataset: OnlineDataset reads from a given corpus_root.
##########################################
class OnlineDataset(Dataset):
    def __init__(self, corpus_root, sp_model_path, max_x_len=300, max_y_len=500, sample_rate=16000):
        self.corpus_root = corpus_root
        self.max_x_len = max_x_len
        self.max_y_len = max_y_len
        self.sample_rate = sample_rate

        # Load SentencePiece model.
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(sp_model_path)
        self.eos_id = self.sp.eos_id()  # End-of-sequence token

        # Collect all samples (audio paths and transcripts) by recursively searching for *.trans.txt files.
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

        # Load audio.
        wave, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            wave = resampler(wave)

        # Extract features (80-dim filterbank) and compute delta and delta-delta.
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


##########################################
# Feature Extraction: compute 80-dim filterbank features, delta, and delta-delta.
##########################################
def extract_features_with_deltas(wave, sample_rate):
    if wave.ndim > 1:
        wave = wave[0]
    fbank = torchaudio.compliance.kaldi.fbank(
        wave.unsqueeze(0),
        num_mel_bins=80,
        sample_frequency=sample_rate
    )
    delta = F.compute_deltas(fbank)
    delta_delta = F.compute_deltas(delta)
    features = torch.stack([fbank, delta, delta_delta], dim=-1)  # (num_frames, 80, 3)
    return features


##########################################
# EMA Update: Update offline model parameters using momentum.
##########################################
def update_offline_model(online_model, offline_model, momentum):
    for online_param, offline_param in zip(online_model.parameters(), offline_model.parameters()):
        offline_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)


##########################################
# Training Loop for Momentum Pseudo-Labeling (Algorithm 1)
##########################################
def train_mpl(labeled_loader, unlabeled_loader, model_online, model_offline, optimizer, ctc_loss_fn, num_epochs, device,
              sp, decoder, eos_id, lambda_u, momentum):
    train_losses = []
    val_losses = []
    val_wers = []
    best_val_loss = torch.inf

    # Log file for training.
    os.makedirs("model/weights/mpl", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    log_file = open("model/weights/mpl/mpl_training_log.txt", "w")

    for epoch in range(1, num_epochs + 1):
        model_online.train()
        model_offline.train()  # offline model is updated via EMA (no grad)
        epoch_loss = 0.0
        num_batches = 0
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        # num_batches is the maximum number of batches between the two loaders.
        num_batches = max(len(labeled_loader), len(unlabeled_loader))

        pbar = tqdm(range(num_batches), desc=f"MPL Epoch {epoch}/{num_epochs}")
        for i in pbar:
            try:
                labeled_batch = next(labeled_iter)
            except StopIteration:
                labeled_batch = None
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_batch = None

            loss_total = 0.0
            loss_lab = 0.0
            loss_unlab = 0.0

            # If a labeled batch is available, compute supervised loss.
            if labeled_batch is not None:
                x_lab, y_lab, x_len_lab, y_len_lab = [b.to(device) for b in labeled_batch]
                logits_lab = model_online(x_lab).transpose(0, 1)    # shape: (T, batch, num_classes)
                log_probs_lab = torch_F.log_softmax(logits_lab, dim=-1)
                x_len_lab_down = compute_downsampled_time(x_len_lab)
                loss_lab = ctc_loss_fn(log_probs_lab, y_lab, x_len_lab_down, y_len_lab)

            if unlabeled_batch is not None:
                x_unlab, _, x_len_unlab, _ = [b.to(device) for b in unlabeled_batch]
                # Use the offline model to generate pseudo-labels.
                logits_unlab_off = model_offline(x_unlab)  # (T, batch, num_classes)
                log_probs_unlab_off = torch_F.log_softmax(logits_unlab_off, dim=-1)
                x_len_unlab_down = compute_downsampled_time(x_len_unlab)
                # Decode using the CTC decoder.
                pseudo_tokens = decoder(log_probs_unlab_off.to(device), x_len_unlab_down.to(device))
                # For each sample, take the best hypothesis.
                pseudo_labels = []
                pseudo_lengths = []
                for beam in pseudo_tokens:
                    best_hypo = beam[0].tokens
                    pseudo_labels.append(best_hypo)
                    pseudo_lengths.append(len(best_hypo))
                max_pl = max(pseudo_lengths) if pseudo_lengths else 0
                pseudo_tensor = torch.full((x_unlab.size(0), max_pl), fill_value=eos_id, dtype=torch.long, device=device)
                for i, seq in enumerate(pseudo_labels):
                    if len(seq) > 0:
                        pseudo_tensor[i, :len(seq)] = torch.tensor(seq, device=device)
                # Compute unsupervised loss on unlabeled data.
                logits_unlab_on = model_online(x_unlab).transpose(0, 1)    # shape: (T, batch, num_classes)
                log_probs_unlab_on = torch_F.log_softmax(logits_unlab_on, dim=-1)
                loss_unlab = ctc_loss_fn(log_probs_unlab_on, pseudo_tensor, x_len_unlab_down,
                                         torch.tensor(pseudo_lengths, dtype=torch.long, device=device))

            # Total loss: supervised + lambda_u * unsupervised.
            total_loss = loss_lab + lambda_u * loss_unlab
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update offline model using EMA.
            update_offline_model(model_online, model_offline, momentum)

            epoch_loss += total_loss.item()
            pbar.set_postfix(loss=f"{total_loss.item():.4f}")

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        log_msg = f"Epoch {epoch} MPL training loss: {avg_loss:.4f}\n"
        print(log_msg, end="")
        log_file.write(log_msg)

        # Evaluate on validation set (using labeled data and WER).
        val_loss, val_wer = evaluate(model_online, val_loader, ctc_loss_fn, device, eos_id, sp, decoder)
        val_losses.append(val_loss)
        val_wers.append(val_wer)
        log_msg = f"Epoch {epoch} validation loss: {val_loss:.4f}, WER: {val_wer * 100:.2f}%\n"
        print(log_msg, end="")
        log_file.write(log_msg)

        # Save checkpoint if improved.
        if val_loss < best_val_loss:
            checkpoint_path = "model/weights/mpl/mpl_model.pt"
            torch.save(model_online.state_dict(), checkpoint_path)
            best_val_loss = val_loss
            log_msg = f"Saved checkpoint to {checkpoint_path}\n"
            print(log_msg, end="")
            log_file.write(log_msg)

        # Save plots (loss and WER).
        plt.figure()
        plt.plot(range(1, epoch + 1), train_losses, label="Train Loss")
        plt.plot(range(1, epoch + 1), val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("CTC Loss")
        plt.title("MPL Training Loss")
        plt.legend()
        plt.savefig("results/mpl_loss.png")
        plt.close()

        plt.figure()
        plt.plot(range(1, epoch + 1), [wer * 100 for wer in val_wers], label="Val WER (%)")
        plt.xlabel("Epoch")
        plt.ylabel("WER (%)")
        plt.title("MPL Validation WER")
        plt.legend()
        plt.savefig("results/mpl_wer.png")
        plt.close()

    log_file.close()
    return train_losses, val_losses, val_wers


##########################################
# Evaluation function using WER via the CTCDecoder.
##########################################
def evaluate(model, loader, ctc_loss_fn, device, eos_id, sp, decoder):
    model.eval()
    total_loss = 0.0
    wer_metric = WordErrorRate()
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            x, y, x_len, y_len = [b.to(device) for b in batch]
            logits = model(x).transpose(0, 1)  # (T, batch, num_classes)
            log_probs = torch_F.log_softmax(logits, dim=-1)
            x_len_down = compute_downsampled_time(x_len)
            loss = ctc_loss_fn(log_probs, y, x_len_down, y_len)
            total_loss += loss.item()
            num_batches += 1

            # Decode predictions.
            log_probs = log_probs.transpose(0, 1).contiguous()  # (batch, T, num_classes)
            predicted_tokens = decoder(log_probs.to(device), x_len_down.to(device))
            # Decode actual and predicted transcripts.
            actual_transcripts = sp.decode([y[i].tolist() for i in range(len(y))])
            predicted_transcripts = sp.decode([predicted_tokens[i][0].tokens for i in range(len(predicted_tokens))])
            wer_metric.update(predicted_transcripts, actual_transcripts)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_wer = wer_metric.compute()
    return avg_loss, avg_wer


##########################################
# Main: set up datasets, models, optimizer, and run MPL training.
##########################################
if __name__ == "__main__":
    # Define separate corpus roots.
    train_corpus_root = "../../data/LibriSpeech/train-clean-100/"  # labeled training data
    unlab_corpus_root = "../../data/LibriSpeech/train-clean-360/"  # unlabeled data
    val_corpus_root = "../../data/LibriSpeech/dev-clean/"  # validation data

    sp_model_path = "spm_model_1k.model"
    max_x_len = 5000
    max_y_len = 500
    batch_size = 25
    num_epochs = 500
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lambda_u = 1.0
    momentum_coef = 0.99954647252

    # Create datasets.
    labeled_dataset = OnlineDataset(train_corpus_root, sp_model_path, max_x_len, max_y_len)
    unlabeled_dataset = OnlineDataset(unlab_corpus_root, sp_model_path, max_x_len, max_y_len)
    val_dataset = OnlineDataset(val_corpus_root, sp_model_path, max_x_len, max_y_len)

    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = 1000 + 1  # 1000 tokens + blank

    # Define the path to the pre-trained weights (.pt file)
    seed_model_path = "model/weights/seed/seed_model_updated.pt"

    # Initialize online and offline models.
    model_online = ASRModel(max_seq_len=max_x_len).to(device)
    model_offline = ASRModel(max_seq_len=max_x_len).to(device)

    # Check if the seed model file exists
    if os.path.isfile(seed_model_path):
        print(f"Loading seed model weights from {seed_model_path}...")
        model_online.load_state_dict(torch.load(seed_model_path, map_location=device))
        model_offline.load_state_dict(model_online.state_dict())  # Synchronize offline model
        print("Seed model weights loaded successfully.")
    else:
        print(f"Seed model file not found at {seed_model_path}. Proceeding with random initialization.")

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model_online.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

    # Load SentencePiece model (for decoding).
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)

    # Build vocabulary list from SentencePiece.
    vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    # Initialize the CTCDecoder using SentencePiece-specific parameters.
    decoder = cuda_ctc_decoder(tokens=vocab, nbest=1, beam_size=20)

    print("CTC Decoder initialized!")
    print("Starting Momentum Pseudo-Labeling training...")
    train_losses, val_losses, val_wers = train_mpl(labeled_loader, unlabeled_loader, model_online, model_offline,
                                                   optimizer, ctc_loss_fn, num_epochs, device, sp, decoder,
                                                   labeled_dataset.sp.eos_id(), lambda_u, momentum_coef)
    print("MPL training complete.")
