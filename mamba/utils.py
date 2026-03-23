import polars as ps
import numpy as np
import torch
from torch import nn, optim
from typing import List
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import itertools

def keep_seq_only(df_path:str):
    dfs = [ps.read_csv(df, separator='\t').drop(['noncodingRNA_name', 'noncodingRNA_fam', 'feature', 'chr', 'start', 'end', 'strand', 'gene_cluster_ID', 'gene_phyloP', 'gene_phastCons']) for df in os.listdir(df_path) if df.endswith('.tsv')]
    all_data = ps.concat(dfs, how='vertical')
    return all_data

def tokenize_DNA(seq: str):
    # Added <cls> token for the start of the sequence
    vocab = { 'A': 1, 'T': 2, 'C': 3, 'G': 4, '<pad>': 0, '<cls>': 5}
    # Add a CLS token at the beginning for classification
    seq_as_list = ['<cls>'] + list(seq)
    tokenized_seq = [vocab.get(token, 0) for token in seq_as_list] # Use .get for safety
    return tokenized_seq

def tokenize_struct(seq: str):
    # Added <cls> token for the start of the sequence
    vocab = {
        '.': 1, '(': 2, ')': 3, '&': 4, '<cls>': 5
    }
    # Add a CLS token at the beginning for classification
    seq_as_list = ['<cls>'] + list(seq)
    tokenized_seq = [vocab.get(token, 0) for token in seq_as_list] # Use .get for safety
    return tokenized_seq

def pad_sequences(sequences, max_len=128, padding_value=0):
    padded = np.full((len(sequences), max_len), padding_value, dtype=np.int64)
    for i, seq in enumerate(sequences):
        length = len(seq)
        if length > max_len:
            padded[i, :] = seq[:max_len]
        else:
            padded[i, :length] = seq
    return padded

def generate_kmers(k: int) -> List[str]:
    bases = ["A", "C", "G", "T"]
    kmers = ["".join(p) for p in itertools.product(bases, repeat=k)]
    return kmers

def build_vocab(k: int):
    kmers = generate_kmers(k)
    vocab = {kmer: idx+1 for idx, kmer in enumerate(kmers)}
    vocab["[CLS]"] = len(vocab) + 1
    return vocab

def separate_cols_chim(chunk):
    seqs_mre = chunk['mre_sequence']
    seqs_mirna = chunk['mirna_sequence']
    labels = chunk['label']
    return seqs_mre, seqs_mirna, labels

def collate_fn_chim(batch):
    chim, labels = zip(*batch)
    chim = pad_sequence(chim, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    return chim, labels

class PairedKmerDataset(Dataset):
    def __init__(self, df, k):
        self.chim, self.labels = separate_cols_chim(df) 
        self.k = k
        self.kmer2idx = build_vocab(k=k) 

    def encode(self, seq):
        kmers = [seq[i:i+self.k] for i in range(len(seq)-self.k+1)]
        return torch.tensor([self.kmer2idx.get(kmer, 0) for kmer in kmers], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.encode(self.chim[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

class PairedKmerDatasetCLS(Dataset):
    def __init__(self, df, k):
        df = df.reset_index(drop=True)
        self.chim, self.labels = separate_cols_chim(df) 
        self.k = k
        self.kmer2idx = build_vocab(k=k) 

    def encode(self, seq):
        kmers = [seq[i:i+self.k] for i in range(len(seq)-self.k+1)]
        return torch.tensor([self.kmer2idx['[CLS]']]+[self.kmer2idx.get(kmer, 0) for kmer in kmers], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.encode(self.chim[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

class DnaModelWithLearnedPE(nn.Module):
    def __init__(self, vocab_size: int = 1025, max_seq_len: int = 64, emb_size: int = 128): 
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.positional_embedding = nn.Embedding(max_seq_len, emb_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        token_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(positions)
        return token_emb + pos_emb

ONEHOT_MAP = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'U': [0, 0, 0, 1],
}

def collate_fn_onehot(batch):
    """Pad variable-length 5-dim one-hot sequences to (batch, max_len, 5)."""
    seqs, labels = zip(*batch)
    seqs   = pad_sequence(seqs, batch_first=True, padding_value=0.0)  # (B, L, 5)
    labels = torch.tensor(labels, dtype=torch.float32)
    return seqs, labels


class OneHotDataset(Dataset):
    """
    Encodes each nucleotide as a 5-dim vector: [A, C, G, T, segment_id].
    segment_id = 0 for mRNA target site positions, 1 for miRNA positions.
    The two subsequences are concatenated in that order.
    Returns tensors of shape (seq_len, 5) and scalar labels.
    """

    def __init__(self, df):
        df = df.reset_index(drop=True) if hasattr(df, 'reset_index') else df
        self.seqs_mre, self.seqs_mirna, self.labels = separate_cols_chim(df)

    def encode(self, seq_mre: str, seq_mirna: str) -> torch.Tensor:
        mre_enc   = [[*ONEHOT_MAP.get(c, [0, 0, 0, 0]), 0] for c in seq_mre]
        mirna_enc = [[*ONEHOT_MAP.get(c, [0, 0, 0, 0]), 1] for c in seq_mirna]
        return torch.tensor(mre_enc + mirna_enc, dtype=torch.float32)  # (L, 5)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.encode(self.seqs_mre[idx], self.seqs_mirna[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class DnaOneHotEncoder(nn.Module):
    """
    Projects per-position one-hot vectors (default dim=5) into model embedding
    space and adds learned positional embeddings.

    Input:  (batch, seq_len, input_dim)  float32
    Output: (batch, seq_len, emb_size)   float32
    """

    def __init__(self, input_dim: int = 5, emb_size: int = 128, max_seq_len: int = 256):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.proj    = nn.Linear(input_dim, emb_size)
        self.pos_emb = nn.Embedding(max_seq_len, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, L)
        return self.proj(x) + self.pos_emb(positions)                     # (B, L, E)

