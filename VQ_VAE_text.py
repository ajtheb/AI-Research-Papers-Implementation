"""
Implementation of VQ-VAE, adapted for text data using PyTorch and BERT tokenizer.
"""

# import spacy
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
from tqdm import tqdm
# import matplotlib.pyplot as plt
import torchvision
from transformers import BertTokenizer

# Load the BERT tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize spaCy tokenizer
# nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(text):
    """Custom tokenizer using spaCy"""
    return tokenizer.tokenize(text)
    # return [token.text.lower() for token in nlp(text) if not token.is_punct and not token.is_space]

def build_vocab(texts, max_size=20000, min_freq=2):
    """Build vocabulary from tokenized texts"""
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    
    vocab = {'<pad>': 0, '<unk>': 1}
    for token, count in counter.most_common(max_size):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab

# class TextDataset(Dataset):
#     def __init__(self, csv_file, tokenizer_fn, vocab):
#         self.data = pd.read_csv(csv_file)
#         self.texts = self.data['Counterspeech'].values
#         self.tokenizer = tokenizer_fn
#         self.vocab = vocab

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         tokens = self.tokenizer(text)
#         numericalized = [self.vocab.get(token, 1) for token in tokens]  # 1 = <unk>
#         return numericalized

from torch.utils.data import Dataset
class TextDataset(Dataset):
    def __init__(self, csv_file, max_length=256):
        self.data = pd.read_csv(csv_file)
        self.texts = self.data['Counterspeech'].values
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def collate_batch(batch):
    """Collate function for BERT tokenizer outputs"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    return {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_masks.to(device)
    }


class TextVQVAE(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, latent_dim=64, num_embeddings=512, commitment_cost=0.25):
        super(TextVQVAE, self).__init__()
        self.emb_dim = emb_dim
        self.commitment_cost = commitment_cost

        # Text embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # Encoder (1D convolutions)
        self.encoder = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, latent_dim, kernel_size=3, padding=1)
        )

        # Vector Quantization
        self.vq_layer = VQEmbedding(num_embeddings, latent_dim, commitment_cost)

        # Decoder (1D transposed convolutions)
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, emb_dim, kernel_size=4, stride=2, padding=1),
            nn.Conv1d(emb_dim, vocab_size, kernel_size=1)
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_len)
        emb = self.embedding(x).permute(0, 2, 1)  # (batch, emb_dim, seq_len)
        
        z_e = self.encoder(emb)
        z_q, vq_loss = self.vq_layer(z_e)
        recon_logits = self.decoder(z_q).permute(0, 2, 1)  # (batch, seq_len, vocab_size)
        
        return recon_logits, vq_loss

class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        # z shape: (batch_size, latent_dim, seq_len)
        batch, dim, seq = z.size()
        z_flat = z.permute(0, 2, 1).contiguous().view(-1, dim)  # (batch*seq, latent_dim)

        # Calculate distances
        distances = (torch.sum(z_flat**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(z_flat, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices).view(batch, seq, dim).permute(0, 2, 1)

        # Commitment loss
        loss = F.mse_loss(z_q, z.detach()) + self.commitment_cost * F.mse_loss(z_q.detach(), z)

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, loss

# Training adjustments
def vqvae_loss(recon_logits, target, vq_loss):
    recon_loss = F.cross_entropy(
        recon_logits.reshape(-1, recon_logits.size(-1)),  # Changed to reshape
        target.view(-1)
    )
    return recon_loss + vq_loss

# Usage example
if __name__ == "__main__":
    # Load data and build vocab
    csv_path_train = "input/Train.csv"
    csv_path_test = "input/Test.csv"
    csv_path_eval = "input/Eval.csv"
    
    train_data = pd.read_csv(csv_path_train)
    test_data = pd.read_csv(csv_path_test)
    eval_data = pd.read_csv(csv_path_eval)
    
    # Create dataset and dataloader
    dataset = TextDataset(csv_path_train)
    train_loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=collate_batch
    )
    
    # Create dataset and dataloader
    dataset = TextDataset(csv_path_test)
    test_loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=collate_batch
    )
    
    # Create dataset and dataloader
    dataset = TextDataset(csv_path_eval)
    eval_loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=collate_batch
    )
    
    vocab_size = tokenizer.vocab_size

    # Test one batch
    sample_batch = next(iter(train_loader))
    sample_batch = sample_batch['input_ids']
    print(f"Batch shape: {sample_batch.shape}")
    # print(tokenize("I will one day"))
    
    
    # Example usage
    model = TextVQVAE(
        vocab_size=vocab_size, 
        emb_dim=256,
        latent_dim=64,
        num_embeddings=512
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    num_epochs = 30  # You can adjust this number based on your needs

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch_input = batch['input_ids']
            inputs = batch_input.to(device)
            
            optimizer.zero_grad()
            recon_logits, vq_loss = model(inputs)
            loss = vqvae_loss(recon_logits, inputs, vq_loss)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        model.eval()
        total_eval_loss = 0
        total_samples = 0
        
        
        with torch.no_grad():
            for batch in eval_loader:
                inputs = batch['input_ids'].to(device)
                
                recon_logits, vq_loss = model(inputs)
                loss = vqvae_loss(recon_logits, inputs, vq_loss)
                
                total_eval_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        avg_eval_loss = total_loss / total_samples

        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(train_loader)}")
        print(f" Average Eval Loss: {avg_eval_loss}")
