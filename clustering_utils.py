#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import torch
from langdetect import detect
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def load_ndc_doc_strings(corpus_dir, filter_english=True, max_docs=None):
    """Load NDC documents from the specified directory"""
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
            with open(os.path.join(corpus_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                if filter_english:
                    try:
                        if detect(text) == 'en':
                            docs.append((filename, text))
                    except:
                        continue
                else:
                    docs.append((filename, text))
                    
            # Stop if we've reached the maximum number of documents
            if max_docs is not None and len(docs) >= max_docs:
                break

    # Convert to DataFrame
    return pd.DataFrame(docs, columns=['file', 'doc'])

def get_embeddings(texts, model, tokenizer, max_length=512):
    """Get embeddings for a list of texts using the given model and tokenizer"""
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

def perform_kmeans_clustering(embeddings, n_clusters=5):
    """Perform K-means clustering on document embeddings"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Calculate silhouette score to evaluate clustering quality
    if len(set(cluster_labels)) > 1:  # Ensure we have more than one cluster
        silhouette = silhouette_score(embeddings, cluster_labels)
    else:
        silhouette = 0
        
    return cluster_labels, kmeans.cluster_centers_, silhouette

def perform_dbscan_clustering(embeddings, eps=0.5, min_samples=3):
    """Perform DBSCAN clustering on document embeddings"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(embeddings)
    
    # Calculate silhouette score if we have more than one cluster and no noise points (-1)
    unique_labels = set(cluster_labels)
    if len(unique_labels) > 1 and not all(l == -1 for l in cluster_labels):
        # Filter out noise points for silhouette calculation
        mask = cluster_labels != -1
        if sum(mask) > 1 and len(set(cluster_labels[mask])) > 1:
            silhouette = silhouette_score(embeddings[mask], cluster_labels[mask])
        else:
            silhouette = 0
    else:
        silhouette = 0
        
    return cluster_labels, silhouette

def find_optimal_k(embeddings, max_k=10):
    """Find optimal number of clusters using silhouette score"""
    silhouette_scores = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        score = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append((k, score))
    
    # Return k with the highest silhouette score
    optimal_k, best_score = max(silhouette_scores, key=lambda x: x[1])
    return optimal_k, best_score, silhouette_scores

def extract_top_terms_per_cluster(docs_df, cluster_labels, n_terms=10):
    """Extract the most common terms for each cluster"""
    docs_df = docs_df.copy()
    docs_df['cluster'] = cluster_labels
    
    stop_words = set(stopwords.words('english'))
    
    cluster_terms = {}
    
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:  # Skip noise points in DBSCAN
            continue
            
        # Combine all texts in the cluster
        cluster_docs = docs_df[docs_df['cluster'] == cluster_id]['doc'].tolist()
        cluster_text = ' '.join(cluster_docs)
        
        # Tokenize and filter out stopwords and punctuation
        words = word_tokenize(cluster_text.lower())
        filtered_words = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 2]
        
        # Find most common terms
        word_counts = Counter(filtered_words).most_common(n_terms)
        cluster_terms[cluster_id] = word_counts
        
    return cluster_terms

def reduce_dimensions_for_visualization(embeddings, n_components=2):
    """Reduce dimensions of embeddings for visualization using PCA"""
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    return reduced_embeddings, pca.explained_variance_ratio_

def get_cluster_summary(docs_df, cluster_labels):
    """Get a summary of documents in each cluster"""
    docs_df = docs_df.copy()
    docs_df['cluster'] = cluster_labels
    
    cluster_summary = {}
    for cluster_id in sorted(set(cluster_labels)):
        cluster_docs = docs_df[docs_df['cluster'] == cluster_id]
        cluster_summary[cluster_id] = {
            'count': len(cluster_docs),
            'files': cluster_docs['file'].tolist()
        }
    
    return cluster_summary
