import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import sklearn
from sklearn.model_selection import train_test_split
from transformers import pipeline
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import torchtext.vocab as vocab_module

df = pd.read_csv("dataset.csv")

classifier = pipeline("sentiment-analysis")
df['sentiment'] = classifier(df['ParaphrasedSubject'].tolist())

df['sentiment_label'] = df['sentiment'].apply(lambda x: x['label'])
df['sentiment_score'] = df['sentiment'].apply(lambda x: x['score'])

label_map = {'NEGATIVE': 0, 'POSITIVE': 1}
df['label'] = df['sentiment_label'].map(label_map)


def tokenize(text):
    return word_tokenize(str(text).lower())
all_tokens = [token for text in df['ParaphrasedSubject'] for token in tokenize(text)]
vocab = Counter(all_tokens) #Vocab is now a dictionary of word counts

word2idx = {'<PAD>': 0, '<UNK>': 1} #Reserve 0 for padding and 1 for unknown words
for word, count in vocab.items(): 
    if count >= 2: 
        word2idx[word] = len(word2idx) #Give each word a unique index
glove = vocab_module.GloVe(name='6B', dim=100) #Load GloVe embeddings (100d), essentially takes the words and maps them to vectors that capture their meanings. This way the model understands meaning without having to also learn from scratch

embedding_matrix = torch.zeros(len(word2idx), 100) #Creates a matrix where each row corresponds to a word in our vocab and each column corresponds to a dimension in the GloVe embedding (100d) 
for word, idx in word2idx.items():
    if word in glove.stoi:
        embedding_matrix[idx] = glove[word]

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=64):
        self.labels = labels
        self.max_len = max_len
        self.encoded = [self.encode(t, word2idx) for t in texts]

    def encode(self, text, word2idx):
        tokens = tokenize(text)
        ids = [word2idx.get(t, 1) for t in tokens]  #Convert tokens to IDs, using 1 for unknown words
        ids = ids[:self.max_len] #Truncate if too long
        ids += [0] * (self.max_len - len(ids))       # Set everything else to 0 for padding
        return ids
    #Every text is tokenized, converted to IDs, truncated/padded to max_len (64), and returned as a list of integers. Every text is the same exact size (64).

    def __len__(self):
        return len(self.labels)   # PyTorch calls this to know dataset size

    def __getitem__(self, idx):   # PyTorch calls this to get one sample
        return (
            torch.tensor(self.encoded[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

texts = df['ParaphrasedSubject'].tolist()
labels = df['label'].tolist()

# Split off test first, then split train into train/val
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.15, random_state=67)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.15, random_state=67)

train_ds = SentimentDataset(X_train, y_train, word2idx)
val_ds   = SentimentDataset(X_val,   y_val,   word2idx)
test_ds  = SentimentDataset(X_test,  y_test,  word2idx)

'''
We split the data into 3 different groups.
Train (~72%) — what the model learns from
Val (~13%) — monitors overfitting during training, drives early stopping
Test (~15%) — touched only once at the end for the final accuracy number
The random state parameter ensures that the splits are the same every time, which is important for reproducibility. 
'''

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True) 
#Train_loader is shuffled to ensure the model doesn't learn any order-based patterns, while val_loader and test_loader are not shuffled since we want consistent evaluation.
val_loader   = DataLoader(val_ds,   batch_size=32)
test_loader  = DataLoader(test_ds,  batch_size=32)


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=2, dropout=0.5, embedding_matrix=None):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0          # tells the model to ignore <PAD> tokens
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,      # input shape: [batch, seq_len, features]
            dropout=dropout,
            bidirectional=True     # reads sequence left-to-right AND right-to-left
        )

        self.dropout = nn.Dropout(dropout) #Our dropout rate is 0.5, so half of the neurons will be randomly turned off during training to prevent overfitting. Neurons cannot be too reliant on any single neuron, so they learn more robust features.

        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 64 features, * 2 because bidirectional
    
        if embedding_matrix is not None: #If we have a pre-trained embedding matrix, we can load it. 
            self.embedding.weight.data.copy_(embedding_matrix)
            self.embedding.weight.requires_grad = False #freezes the vectors so they don't get updated during training. This way we keep the semantic information from GloVe intact, and the model learns to use those fixed embeddings to make predictions without altering them.


    def forward(self, x):
        # x shape: [batch_size, seq_len]

        embedded = self.embedding(x) #Each integer ID gets its 100-dimensional GloVe vector, so the shape becomes [batch_size, seq_len, embed_dim]

        output, (hidden, cell) = self.lstm(embedded)
        '''
        Three things are returned: output, hidden, and cell.
        Output [batch_size, seq_len, hidden_dim * 2]: This is what the LSTM outputs for each token, it holds the forward and backward runs. However, we don't need this for sentiment analysis.
        Hidden [num_layers * 2, batch_size, hidden_dim]: This is the final hidden state for each layer and direction. The last two entries (hidden[-2] and hidden[-1]) correspond to the last 
        forward and backward layers, which have the most context, and infomration, which we will use for classification.
        Cell [num_layers * 2, batch_size, hidden_dim]: Cell is longer term memory that helps the LSTM decide what to keep or forget.
        '''

        # grab the final forward and backward hidden states and concatenate
        hidden_fwd = hidden[-2, :, :]   # last forward layer
        hidden_bwd = hidden[-1, :, :]   # last backward layer
        combined = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        # combined shape: [batch_size, hidden_dim * 2]

        dropped = self.dropout(combined) #Applies our dropout 

        return self.fc(dropped) #Returns two logits, one for each class, the higher logit is the predicted class.

def predict_sentiment(model, text, word2idx, max_len=64, device='cpu'):
    model.eval() #Turns on inference mode, which turns off dropout and other training-specific layers. This ensures we get consistent predictions.
    model.to(device) 
    
    # Tokenize and encode
    tokens = tokenize(str(text))
    ids = [word2idx.get(t, 1) for t in tokens]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    
    # Convert to tensor and add batch dimension
    tensor = torch.tensor([ids], dtype=torch.long).to(device)
    
    # Predict
    with torch.no_grad(): # We don't want to calculate gradients during inference, so we wrap it in torch.no_grad() to save memory and computation.
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0] # We need to convert our logits to probabilities to get confidence scores, and we take the first (and only) item in the batch with [0].
        pred_label = torch.argmax(probs).item() # Get the index of the highest probability, which corresponds to the predicted class (0 or 1).
    
    return {
        'text': text,
        'prediction': idx2label[pred_label],
        'confidence': probs[pred_label].item(),
        'probabilities': {
        'NEGATIVE': probs[0].item(),
        'POSITIVE': probs[1].item()}
    }
    # Return the text, our prediction, the confidence we have in our precition and the probability it is one or the other class.
 

def run_training(model, train_loader, val_loader, n_epochs=30, device='cpu', patience=3):
    """Train the model and track best validation performance."""
    
    model.to(device) # Move the model to whatever device we are using.
    criterion = nn.CrossEntropyLoss() # We have to measure our loss compared to the true labels using CrossEntropyLoss 
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.5
)
    '''
    Adam is an optimization algorithm that adjusts the learning rate for each parameter based on estimates of first and second moments of the gradients. 
    We use it here with a learning rate of 0.0005 and a weight decay of 1e-4 to prevent overfitting by adding a penalty to large weights.
    Our lambda function prevents the GloVe embeddings from getting updates (we froze them).
    '''
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    
    #Training loop
    for epoch in range(n_epochs):
        model.train() #In training mode, so our dropouts and other training-specific layers are active. This allows the model to learn more robust features.
        train_loss = 0 # losses are accumulated, so after each epoch, reset the train_loss to 0 to start fresh for the next epoch.
        for batch_texts, batch_labels in train_loader:
            batch_texts = batch_texts.to(device) 
            batch_labels = batch_labels.to(device)
            #Moves our tensors to the cpu/gpu so the model can process them.

            logits = model(batch_texts) #Forward pass to get our logits
            loss = criterion(logits, batch_labels) #Our cross-entropy loss: how far off are we from the true labels?
            
            optimizer.zero_grad() #Clear gradients from the previous step to prevent accumulation
            loss.backward() #Backpropagation to compute gradients of each parameter with respect to the loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #Clips the gradients to a maximum norm of 1.0 to prevent exploding gradients, which can destabilize training.
            optimizer.step() #Updates the model parameters based on the computed gradients and the optimization algorithm.
            
            train_loss += loss.item() #Adds the loss to our train_loss accumulator to keep track of the total loss for the epoch.
        
        train_loss /= len(train_loader) #averages all the losses over the number of batches to get the average loss for the epoch, which is a more stable metric to track than the total loss.
        train_losses.append(train_loss) # Appends the average for the epoch to our list of train losses for later visualization.
        
        # Validation
        model.eval() # Evalution mode: Dropouts are off
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(): #We aren't updating any weights, so we don't need to calculate gradients during validation, which saves memory and computation.
            for batch_texts, batch_labels in val_loader:
                #Same as before, we get our logits and calculate our loss, but we also want to track accuracy on the validation set to see how well our model is doing.

                batch_texts = batch_texts.to(device)
                batch_labels = batch_labels.to(device)
                
                logits = model(batch_texts)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item()


                preds = torch.argmax(logits, dim=1) #Get the predicted class by taking the index of the highest logit for each sample in the batch.
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_labels.size(0) #Counts correct predictions and total samples across all validation batches.
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        #Computes average loss over the epoch

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step(val_loss) # We make a schdeduler, if our valdiation loss doesn't improve, we might be taking too big of steps, so we will reduce the learning rate. This helps the model converge to a better minimum by allowing it to take smaller steps when it's not improving.

        # Early stop
        if val_loss < best_val_loss:
            #If this is the best val loss seen so far, snapshot the model weights and reset the no-improvement counter.
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            #Otherwise increment the counter. If it hits patience=3, print and break out of the epoch loop entirely.
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_model_state) #Restores the best weights captured during training, if the final epoch was overfit, then we will revert back to the best weights from an earlier epoch that had better validation performance.
    print(f"\nLoaded best model (Val Loss: {best_val_loss:.4f})")
    
    return model, train_losses, val_losses #Returns the trained model and loss histories. Loss lists are used for plotting training curves.

idx2label = {0: 'NEGATIVE', 1: 'POSITIVE'} # Everything our model outputs is 0s and 1s, we need a map to convert it back into Negative and Positive
if __name__ == "__main__": #Only run if I'm running it directly
    # Device setup - Choose cuda, then mps, then cpu
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}\n")
    
    # Hyperparameters
    VOCAB_SIZE = len(word2idx)
    EMBED_DIM = 100 # This needs to be 100 to match our GloVe embeddings, if we change this, we need to change the dimension of the GloVe vectors we load as well.
    HIDDEN_DIM = 64
    OUTPUT_DIM = 2
    N_LAYERS = 2
    DROPOUT = 0.5
    
    model = SentimentLSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT, embedding_matrix=embedding_matrix)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Train
    model, train_losses, val_losses = run_training(
        model, train_loader, val_loader, n_epochs=30, device=device
    )
    
    # Evaluate on test set
    print()
    evaluate_model(model, test_loader, device=device)



