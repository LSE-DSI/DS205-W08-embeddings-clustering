#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from lets_plot import *
from lets_plot.mapping import as_discrete

def prepare_document_chunks(doc_text, filename):
    """
    Split a document into line-by-line chunks using the same strategy as in NB02
    
    Parameters:
    -----------
    doc_text : str
        The text of the document to be chunked
    filename : str
        The filename of the document
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing text chunks with their line numbers and file information
    """
    # Split the document by newlines
    lines = doc_text.split('\n')
    
    # Create a DataFrame with a row for each line
    chunks_df = pd.DataFrame({
        'text': lines,
        'line_number': range(1, len(lines) + 1),
        'file': filename
    })
    
    # Filter out empty lines but keep the line numbering
    chunks_df = chunks_df[chunks_df['text'].str.strip() != '']
    
    return chunks_df

def create_interactive_plot(viz_df, color_column, title, explained_variance):
    """
    Create an interactive scatter plot with tooltips showing the text of document chunks
    
    Parameters:
    -----------
    viz_df : pd.DataFrame
        DataFrame containing visualization data with x, y coordinates, cluster assignments, and text
    color_column : str
        The name of the column to use for coloring the points
    title : str
        The title of the plot
    explained_variance : list
        List containing explained variance ratios for the principal components
    
    Returns:
    --------
    LetsPlot plot object
        The interactive plot with tooltips
    """
    # Create tooltips for the interactive plot
    tooltips = [
        "text: @{text|n}",
        "line: @{line_number}",
        "cluster: @{" + color_column + "}"
    ]
    
    # Create the interactive scatter plot
    plot = ggplot(viz_df, aes(x='x', y='y', color=as_discrete(color_column))) + \
        geom_point(size=4, alpha=0.7) + \
        labs(title=title,
             x=f"PC1 ({explained_variance[0]:.1%} variance)", 
             y=f"PC2 ({explained_variance[1]:.1%} variance)") + \
        theme_minimal()
    
    return plot

def create_section_flow_visualization(section_df, colors_dict):
    """
    Create a horizontal bar chart showing document flow of topics/sections
    
    Parameters:
    -----------
    section_df : pd.DataFrame
        DataFrame containing information about document sections
    colors_dict : dict
        Dictionary mapping cluster IDs to colors
        
    Returns:
    --------
    LetsPlot plot object
        The section flow visualization
    """
    # Create a color scale for different clusters
    section_df['width'] = section_df['end_pos'] - section_df['start_pos']
    
    # Create tooltips for showing section information
    tooltips = [
        "Cluster: @{cluster}",
        "Lines: @{start_line}-@{end_line}",
        "Length: @{length} lines"
    ]
    
    # Create the section flow visualization
    flow_plot = ggplot(section_df, aes(x='start_pos', y=0)) + \
        geom_segment(aes(x='start_pos', xend='end_pos', y=0, yend=0, color='factor(cluster)'), 
                     size=20, alpha=0.8) + \
        labs(title="Document Structure - Topic Flow", 
             x="Position in Document", 
             y="") + \
        scale_x_continuous(labels=['Start', '25%', '50%', '75%', 'End'], 
                          breaks=[0, 0.25, 0.5, 0.75, 1]) + \
        theme_minimal() + \
        theme(legend_title='Cluster', axis_text_y=element_blank(), axis_ticks_y=element_blank())
    
    return flow_plot
