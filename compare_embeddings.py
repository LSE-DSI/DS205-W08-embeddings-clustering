#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compare ClimateBERT and DistilRoBERTa embeddings
Explores which model better captures climate-specific language
"""

import os
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from tqdm.notebook import trange
import torch

from lets_plot import *
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_embeddings

LetsPlot.setup_html()

def load_models():
    """Load both models for comparison"""
    print("Loading models...")
    
    # ClimateBERT model
    climate_model_dir = "./local_models/climatebert/distilroberta-base-climate-f"
    climate_tokenizer = AutoTokenizer.from_pretrained(climate_model_dir)
    climate_model = AutoModel.from_pretrained(climate_model_dir)
    
    # DistilRoBERTa model (general purpose)
    roberta_model_name = "distilroberta-base"
    roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
    roberta_model = AutoModel.from_pretrained(roberta_model_name)
    
    return (climate_tokenizer, climate_model), (roberta_tokenizer, roberta_model)

def compare_climate_terms(models, climate_terms, general_terms):
    """Compare how each model handles climate-specific vs general terms"""
    climate_tokenizer, climate_model = models[0]
    roberta_tokenizer, roberta_model = models[1]
    
    # Combine all terms for comparison
    all_terms = climate_terms + general_terms
    labels = ["climate"] * len(climate_terms) + ["general"] * len(general_terms)
    
    # Get embeddings from each model
    climate_embeddings = get_embeddings(all_terms, climate_model, climate_tokenizer)
    roberta_embeddings = get_embeddings(all_terms, roberta_model, roberta_tokenizer)
    
    # Calculate similarity matrices for each model
    climate_sim = cosine_similarity(climate_embeddings)
    roberta_sim = cosine_similarity(roberta_embeddings)
    
    # Create DataFrames for similarity results
    climate_df = pd.DataFrame(climate_sim, index=all_terms, columns=all_terms)
    roberta_df = pd.DataFrame(roberta_sim, index=all_terms, columns=all_terms)
    
    # Add metadata for term categories
    result_df = pd.DataFrame({
        'term': all_terms,
        'category': labels,
        'climate_within_category_avg': 0.0,
        'roberta_within_category_avg': 0.0,
    })
    
    # Calculate within-category similarity averages
    for category in ['climate', 'general']:
        category_terms = [t for i, t in enumerate(all_terms) if labels[i] == category]
        for term in category_terms:
            # For each term, calculate average similarity with others in same category
            other_terms = [t for t in category_terms if t != term]
            result_df.loc[result_df['term'] == term, 'climate_within_category_avg'] = \
                climate_df.loc[term, other_terms].mean()
            result_df.loc[result_df['term'] == term, 'roberta_within_category_avg'] = \
                roberta_df.loc[term, other_terms].mean()
    
    # Calculate difference (how much more ClimateBERT captures within-category similarity)
    result_df['similarity_difference'] = result_df['climate_within_category_avg'] - result_df['roberta_within_category_avg']
    
    return result_df, climate_df, roberta_df

def visualize_comparison(result_df):
    """Create visualization with lets_plot"""
    # Convert to long format for easier plotting
    plot_df = pd.melt(
        result_df, 
        id_vars=['term', 'category'],
        value_vars=['climate_within_category_avg', 'roberta_within_category_avg'],
        var_name='model', 
        value_name='similarity'
    )
    
    # Map model names to cleaner labels
    plot_df['model'] = plot_df['model'].map({
        'climate_within_category_avg': 'ClimateBERT', 
        'roberta_within_category_avg': 'DistilRoBERTa'
    })
    
    # Create grouped bar chart
    p = ggplot(plot_df, aes(x='term', y='similarity', fill='model')) + \
        geom_bar(stat='identity', position='dodge') + \
        facet_wrap('category') + \
        ggtitle("Within-Category Similarity: ClimateBERT vs DistilRoBERTa") + \
        theme_classic() + \
        theme(axis_text_x=element_text(angle=45, hjust=1))
    
    # Create similarity difference plot
    diff_p = ggplot(result_df, aes(x='term', y='similarity_difference', fill='category')) + \
        geom_bar(stat='identity') + \
        ggtitle("ClimateBERT Advantage in Similarity (positive = ClimateBERT better)") + \
        theme_classic() + \
        theme(axis_text_x=element_text(angle=45, hjust=1))
    
    return p, diff_p

def run_comparison():
    """Main function to run the comparison"""
    # Load models
    models = load_models()
    
    # Define climate-specific and general terms for comparison
    climate_terms = [
        "carbon emissions",
        "climate change adaptation",
        "greenhouse gas reduction",
        "renewable energy transition",
        "nationally determined contributions",
        "climate resilience",
        "carbon neutrality",
        "climate finance",
        "loss and damage",
        "net zero emissions"
    ]
    
    general_terms = [
        "international cooperation",
        "policy implementation",
        "sustainable development",
        "economic growth",
        "resource allocation",
        "governance structure",
        "technical assistance",
        "capacity building",
        "monitoring framework",
        "global partnership"
    ]
    
    # Run comparison
    result_df, climate_sim, roberta_sim = compare_climate_terms(models, climate_terms, general_terms)
    
    # Display results
    print("\n--- Comparison Results ---")
    print("Average within-category similarity for climate terms:")
    climate_terms_df = result_df[result_df['category'] == 'climate']
    display(climate_terms_df[['term', 'climate_within_category_avg', 'roberta_within_category_avg']]
            .sort_values('climate_within_category_avg', ascending=False)
            .style.format({'climate_within_category_avg': '{:.4f}', 'roberta_within_category_avg': '{:.4f}'})
            .set_caption('Climate Terms Similarity'))
    
    print("\nAverage within-category similarity for general terms:")
    general_terms_df = result_df[result_df['category'] == 'general']
    display(general_terms_df[['term', 'climate_within_category_avg', 'roberta_within_category_avg']]
            .sort_values('roberta_within_category_avg', ascending=False)
            .style.format({'climate_within_category_avg': '{:.4f}', 'roberta_within_category_avg': '{:.4f}'})
            .set_caption('General Terms Similarity'))
    
    # Create and display visualizations
    p, diff_p = visualize_comparison(result_df)
    display(p)
    display(diff_p)
    
    # Calculate overall averages
    climate_avg_climate = climate_terms_df['climate_within_category_avg'].mean()
    climate_avg_general = general_terms_df['climate_within_category_avg'].mean()
    roberta_avg_climate = climate_terms_df['roberta_within_category_avg'].mean()
    roberta_avg_general = general_terms_df['roberta_within_category_avg'].mean()
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'Model': ['ClimateBERT', 'DistilRoBERTa'],
        'Climate Terms Coherence': [climate_avg_climate, roberta_avg_climate],
        'General Terms Coherence': [climate_avg_general, roberta_avg_general],
        'Climate Specialization Ratio': [climate_avg_climate/climate_avg_general, 
                                        roberta_avg_climate/roberta_avg_general]
    })
    
    print("\n--- Summary Statistics ---")
    display(summary_df.style
            .format({
                'Climate Terms Coherence': '{:.4f}', 
                'General Terms Coherence': '{:.4f}',
                'Climate Specialization Ratio': '{:.4f}'
            })
            .set_caption('Model Comparison Summary'))
    
    # Analyse and display conclusions
    climate_advantage = climate_avg_climate - roberta_avg_climate
    
    print("\n--- Conclusions ---")
    print(f"ClimateBERT has a {climate_advantage:.4f} advantage in capturing similarity between climate terms.")
    
    if climate_advantage > 0:
        print("ClimateBERT better captures the semantic relationships between climate-specific terms.")
        if climate_avg_general < roberta_avg_general:
            print("ClimateBERT is more specialized for climate language, while DistilRoBERTa has better general language understanding.")
        else:
            print("ClimateBERT shows better performance on both climate and general terms.")
    else:
        print("Surprisingly, DistilRoBERTa better captures the semantic relationships between climate-specific terms.")

    print("\nThis suggests that:")
    if climate_advantage > 0:
        print("✅ Domain-specific training improves a model's ability to capture nuanced relationships in climate terminology.")
        print("✅ When analyzing climate policy documents, ClimateBERT would likely provide more accurate semantic understanding.")
    else:
        print("❓ The domain-specific training may not have significantly improved climate language understanding.")
        print("❓ General-purpose models may be sufficient for basic climate policy document analysis.")

if __name__ == "__main__":
    run_comparison()
