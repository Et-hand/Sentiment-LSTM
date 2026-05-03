# Sentiment-LSTM
A bidirectional LSTM for binary sentiment classification of financial news, achieving 86.2% test accuracy and 0.85 F1 on Saudi Arabian financial headlines

## Project Overview
This project's goal was to classify financial news as either "positive" or "negative". The dataset contains Saudi Arabian Financial News, and I used HuggingFace's Sentiment Analysis to classify them as my benchmark.


exploration.ipynb - My notebook that documents more of my thinking process than train.py

train.py - A script that can be directly run.

## Model Architecture

A bidirectional 2-layer LSTM trained for binary sentiment classification (POSITIVE / NEGATIVE).

### Pipeline
```
Input Text → Tokenization (NLTK) → Integer Encoding → GloVe-100 Embeddings (frozen)
→ Bidirectional LSTM (2 layers) → Dropout (0.5) → Linear Classifier → Predicted Class
```

### Layers
| Layer | Details |
|---|---|
| Embedding | GloVe-6B-100d pretrained, frozen, vocab-sized lookup table |
| LSTM | 2 layers, hidden dim 64, bidirectional (effective dim 128) |
| Dropout | Rate 0.5, applied to final hidden state before classifier |
| Linear | 128 → 2 (NEGATIVE / POSITIVE) |

### Training Details
| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=0.0005, weight_decay=1e-4) |
| Loss | CrossEntropyLoss |
| LR Scheduler | ReduceLROnPlateau (patience=2, factor=0.5) |
| Early Stopping | Patience=3 epochs |
| Max Epochs | 30 |
| Batch Size | 32 |

### Results
| Split | Value |
|---|---|
| Validation Loss | 0.3193 |
| Test Accuracy | 86.23% |

## Engineering Decisions:
### Neutral
I initially had my code classify between positive, negative, and neutral. I realized later that HuggingFace only classifies data as positive or negative. I'll be sure to fully look through my transformed data and make sure its consistent to prevent careless mistakes. 

### Tokenization
My initial tokenization looked like: 
```python
text = str(text).lower()
```
Which I changed for a pre-made tokenizer:
```python
nltk.word_tokenize
```
### Embeddings
I had the embeddings be made from scratch every time, which was not as accurate as GloVe, which I ended up using, because I couldn't capture the full range of the language with random short summaries.

### Hyperparameters
HIDDEN_DIM was set to 128. The small database did not need that many parameters, so I turned it down to 64, which saved time and memory and had no significant impact on performance. 

### The Overfitting Problem
This was the most complex problem: my outputs would look like this:
| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 5     | 0.24       | 0.41     |
| 10    | 0.09       | 0.57     |

This was a clear signal that it was overfitting; the training set was getting better, but the validation loss would increase. The gap between the training loss and validation loss was getting larger. 

#### Dropout
I increased dropout from 0.3 to 0.5, which increased my accuracy by ~1-2%. This was a larger value than I had initially thought appropriate. This reduced overfitting by making the nodes less dependent on each other.
#### Early Stops
The gap between the training and validation loss was growing, so if two more epochs passed without a better validation loss, then we would terminate the epochs to prevent overfitting and save compute.
#### Learning Rate
Learning rate was initially set at 0.001, but part of the overfitting could have been due to overshooting the loss, and so reducing the rate was effective. 
#### Scheduler
The scheduler lowers the learning rate when the loss plateaus; this is the best of both worlds. I can have quick run-throughs in the beginning, but as the error gets smaller and smaller, I can reduce the learning rate to get a more accurate result.
#### Weight Decay
Large weights will overinfluence other nodes and create disruptions, which is common in overfitting, so normalizing all the weights to be under 1.0 also helped.
#### Gradient Clipping
This prevents the exploding gradient problem by setting a cap during backprop, so it prevents too much overfitting. 

## Confusion Matrix
<img width="900" height="750" alt="image" src="https://github.com/user-attachments/assets/303ccc3f-d8e3-456d-a798-bedeb29a0234" />
