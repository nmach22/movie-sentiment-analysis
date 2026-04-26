# Movie Sentiment Analysis

A PyTorch-based sentiment classification project built on the [Kaggle Sentiment Analysis on Movie Reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews) competition dataset.

## Task

5-class sentiment classification of movie review phrases:

| Label | Sentiment |
|-------|-----------|
| 0 | Negative |
| 1 | Somewhat Negative |
| 2 | Neutral |
| 3 | Somewhat Positive |
| 4 | Positive |

**Dataset split:** 124,848 training samples / 31,212 validation samples

## Preprocessing

- Lowercasing and removal of non-alphanumeric characters
- Custom vocabulary built from training data (minimum frequency threshold: 2)
- Special tokens: `<PAD>` (idx 0), `<UNK>` (idx 1)
- Padded sequences with `pack_padded_sequence` for efficient RNN processing

---

## Models

### 1. SentimentRNN

A baseline vanilla RNN model.

- **Architecture:** Embedding → RNN → Linear classifier
- **Classification:** Uses the last hidden state of the final RNN layer
- **Purpose:** Establishes a simple baseline; straightforward to swap the `nn.RNN` cell for `nn.LSTM` or `nn.GRU`

---

### 2. LSTM

A straightforward single-direction LSTM.

- **Architecture:** Embedding → LSTM (with dropout 0.3) → Linear classifier
- **Classification:** Uses the last hidden state of the final LSTM layer
- **Purpose:** Baseline improvement over the vanilla RNN, capturing longer-range dependencies

---

### 3. SentimentLSTM ⭐ Best Submission

A bidirectional LSTM with weight initialization — the best performing model on the Kaggle leaderboard despite having a simpler classifier head than later models.

- **Architecture:** Embedding → BiLSTM (2 layers, dropout 0.3) → Dropout → Linear classifier
- **Embedding dim:** 128 | **Hidden dim:** 256 | **Output:** 512 (bidirectional)
- **Learnable parameters:** ~4.5 million
- **Classification:** Concatenates the last forward and backward hidden states
- **Notable:** Uniform embedding initialization (`±0.1`), PAD embedding zeroed out explicitly; gracefully falls back to inferred lengths when not provided explicitly

**Results:**

| Split | Accuracy |
|-------|----------|
| Train (Epoch 21) | 0.7509 |
| Validation | 0.6653 |
| Public score | matching private |
| Private score | matching public |

The train/val accuracy gap (~0.085) was the tightest of all models, indicating better generalization. Notably, its public and private leaderboard scores were identical — unlike other models — suggesting it was the most robust and least overfit to the public test split.

#### Checkpoints

The best model checkpoint (`SentimentLSTM`) is available on [Google Drive](https://drive.google.com/drive/folders/1nxlUfJrPnzOwMZddkuQW_zkq_r64YmGG?usp=drive_link).
Alternatively, it can be reproduced with the config above in ~21 epochs on a T4 GPU.

---

### 4. PureBiLSTM

A bidirectional LSTM with an attention mechanism.

- **Architecture:** Embedding → BiLSTM (2 layers, dropout 0.3) → Attention → MLP classifier head
- **Embedding dim:** 128 | **Hidden dim:** 256 | **Output:** 512 (bidirectional)
- **Attention:** A learned linear layer scores each time step; a softmax produces weights; the context vector is the weighted sum of all hidden states — giving the model focus over the full sequence rather than just the final state
- **Classifier head:** `Linear(512 → 256) → ReLU → Dropout → Linear(256 → 5)`
- **Validation accuracy: ~65.11%**

---

### 5. EnhancedSentimentLSTM

An upgraded bidirectional LSTM with a wider embedding and attention.

- **Architecture:** Embedding → BiLSTM (2 layers, dropout 0.4) → Attention → MLP classifier head
- **Embedding dim:** 300 (GloVe-compatible) | **Hidden dim:** 256 | **Output:** 512 (bidirectional)
- **Attention:** Same soft-attention design as `PureBiLSTM` (tanh scoring → softmax weights → weighted sum)
- **Classifier head:** `Linear(512 → 256) → ReLU → Dropout → Linear(256 → 5)`
- **Notable:** The 300-dimensional embedding is designed to be compatible with pretrained GloVe vectors for potential further improvement

---

## Training Setup

- **Optimizer:** Adam
- **Loss:** CrossEntropyLoss
- **Early stopping:** Patience of 5 epochs on validation accuracy
- **Hardware:** Google Colab T4 GPU

## Environment

```
Python 3
PyTorch
pandas, numpy, scikit-learn
Google Colab
```
