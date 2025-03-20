# Momentum Pseudo-Labeling (MPL) Project

This repository contains an implementation of the paper "Momentum Pseudo-Labeling: Semi-Supervised ASR With Continuously Improving Pseudo-Labels" by Higuchi et al.

---

## 1. Building the Vocabulary

Run the following command to build a 1000-token vocabulary from the LibriSpeech dev-clean dataset:

```bash
python build_vocab.py
```

## 2. Testing the Vocabulary
Test the functionality of the newly created vocabulary on some sample sentences by running:

```bash
python test_vocab.py
```

## 3. Training the Seed Model
To train the seed model, run:

```bash
python train_seed.py
```

## 4. Training with Momentum Pseudo-Labeling
To start training using Momentum Pseudo-Labeling, run:

```bash
python train_mpl.py
```

## 5. Testing a Model

To test a trained model (either seed or MPL) on the test set, run:

```bash
python test_model.py
```

- **Important**: You should set the variable model_path in the script to the path of the model weights you want to load and test.

---

## Outputs and Logs

- All training loss and WER curves will be saved to the ```/results``` folder.
- All model weights and training logs will be saved to the corresponding folder under ```/model/weights```.
