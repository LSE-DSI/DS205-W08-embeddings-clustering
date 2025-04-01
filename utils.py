#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import torch
from langdetect import detect

def load_ndc_doc_strings(corpus_dir, filter_english=True, max_docs=None):
    """
    Load NDC documents from the specified directory
    
    Parameters:
    -----------
    corpus_dir : str
        Path to the directory containing text documents
    filter_english : bool, optional
        Whether to only keep English documents using language detection
    max_docs : int, optional
        Maximum number of documents to load (for testing or sample analysis)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns 'file' and 'doc' containing filenames and document texts
    """
    print(f"Loading NDC documents from {corpus_dir}")

    # Check if corpus directory exists
    if not os.path.exists(corpus_dir):
        print(f"Corpus directory not found: {corpus_dir}")
        return pd.DataFrame(columns=['file', 'doc'])

    # List of tuples to store (filename, document content)
    docs = []

    # Read all text files in the directory
    for filename in os.listdir(corpus_dir):
        if filename.endswith('.txt'):
            try:
                with open(os.path.join(corpus_dir, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    if filter_english:
                        try:
                            if detect(text[:1000]) == 'en':  # Just check beginning for efficiency
                                docs.append((filename, text))
                        except:
                            # If language detection fails, include it anyway
                            docs.append((filename, text))
                    else:
                        docs.append((filename, text))
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
            # Stop if we've reached the maximum number of documents
            if max_docs is not None and len(docs) >= max_docs:
                break

    # Convert to DataFrame
    return pd.DataFrame(docs, columns=['file', 'doc'])

def get_embeddings(texts, model, tokenizer, max_length=512):
    """
    Get embeddings for a list of texts using the given model and tokenizer
    
    Parameters:
    -----------
    texts : list
        List of text strings to generate embeddings for
    model : transformers.PreTrainedModel
        The transformer model used to generate embeddings
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer for the model
    max_length : int, optional
        Maximum sequence length for tokenization
        
    Returns:
    --------
    np.ndarray
        Array of embeddings with shape (len(texts), embedding_dim)
    """
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for text in texts:
            # Tokenize and truncate
            inputs = tokenizer(
                text, 
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get model outputs
            outputs = model(**inputs)
            
            # Use CLS token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(embedding[0])
            
    return np.array(embeddings)

def cosine_similarity_pairwise(X, Y=None):
    """
    Compute pairwise cosine similarity between the rows of X and Y
    
    Parameters:
    -----------
    X : np.ndarray
        Matrix of shape (n_samples_X, n_features)
    Y : np.ndarray, optional
        Matrix of shape (n_samples_Y, n_features)
        
    Returns:
    --------
    np.ndarray
        Similarity matrix of shape (n_samples_X, n_samples_Y) if Y is provided,
        or (n_samples_X, n_samples_X) if Y is None
    """
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    if Y is None:
        Y_normalized = X_normalized
    else:
        Y_normalized = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        
    similarities = np.dot(X_normalized, Y_normalized.T)
    return similarities