"""
evaluate.py
Evaluation metrics, confusion matrices, model comparison plots, and EDA charts.
"""

import os
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from wordcloud import WordCloud

# Set visual styling defaults
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def calculate_metrics(y_true, y_pred):
    """
    Compute accuracy, weighted precision, weighted recall, and weighted F1 score.

    Returns:
        dict: Summary of metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, digits=4, zero_division=0)

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'report_dict': report_dict,
        'report_str': report_str
    }


def plot_confusion_matrices(model_results_dict, labels, save_path):
    """
    Plot and save confusion matrices for all evaluated models in a clean multi-panel figure.

    Parameters:
        model_results_dict (dict): {model_name: (y_true, y_pred)}
        labels (list): Ordered list of class labels.
        save_path (str): Filepath to save the plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_models = len(model_results_dict)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5), dpi=300)

    if num_models == 1:
        axes = [axes]

    palettes = ['Blues', 'Greens', 'Purples', 'Oranges']

    for idx, (name, (y_true, y_pred)) in enumerate(model_results_dict.items()):
        ax = axes[idx]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cmap = palettes[idx % len(palettes)]

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap=cmap,
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=1.2,
            linecolor='white',
            annot_kws={"size": 13, "weight": "bold"}
        )
        ax.set_title(f'{name}\nConfusion Matrix', fontsize=13, weight='bold', pad=12)
        ax.set_xlabel('Predicted Label', fontsize=11, weight='bold')
        ax.set_ylabel('True Label', fontsize=11, weight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrices to: {save_path}")


def plot_model_comparison(results_df, save_path):
    """
    Plot a multi-metric comparison bar chart and save to file.

    Parameters:
        results_df (pd.DataFrame): DataFrame with columns ['Model', 'Accuracy', 'Precision', 'Recall', 'F1']
        save_path (str): Filepath to save figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Melt dataframe for easy seaborn grouped barplotting
    metrics = [c for c in ['Accuracy', 'Precision', 'Recall', 'F1'] if c in results_df.columns]
    melted = results_df.melt(id_vars=['Model'], value_vars=metrics, var_name='Metric', value_name='Score')

    plt.figure(figsize=(10, 6), dpi=300)
    palette = sns.color_palette(['#4361EE', '#3A0CA3', '#7209B7', '#4CC9F0'])

    ax = sns.barplot(
        data=melted,
        x='Model',
        y='Score',
        hue='Metric',
        palette=palette,
        edgecolor='black',
        linewidth=0.8
    )

    plt.title('Machine Learning Models Performance Comparison', fontsize=15, weight='bold', pad=15)
    plt.xlabel('Algorithm', fontsize=12, weight='bold')
    plt.ylabel('Score (0.0 - 1.0)', fontsize=12, weight='bold')
    plt.ylim(0.0, 1.08)
    plt.legend(title='Metric', loc='lower right', frameon=True, shadow=True)

    # Annotate bar heights
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f'{height:.3f}',
                (p.get_x() + p.get_width() / 2., height),
                ha='center',
                va='bottom',
                fontsize=8.5,
                rotation=0,
                xytext=(0, 3),
                textcoords='offset points',
                weight='semibold'
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved model comparison plot to: {save_path}")


def generate_eda_visualizations(df, text_col='review', label_col='sentiment', output_dir='outputs'):
    """
    Generate and save all required EDA visualizations:
    1. Sentiment countplot
    2. Review length distribution histogram
    3. Top 15 words per class bar chart
    4. Word clouds per sentiment class
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure length columns exist
    df_eda = df.copy()
    df_eda['word_count'] = df_eda[text_col].astype(str).apply(lambda x: len(x.split()))
    df_eda['char_count'] = df_eda[text_col].astype(str).apply(len)

    # Order of sentiments
    sentiment_order = ['positive', 'neutral', 'negative']
    present_order = [s for s in sentiment_order if s in df_eda[label_col].unique()]
    palette_map = {'positive': '#2EC4B6', 'neutral': '#FF9F1C', 'negative': '#E71D36'}

    # 1. Sentiment Countplot
    plt.figure(figsize=(7, 5), dpi=300)
    ax = sns.countplot(
        data=df_eda,
        x=label_col,
        order=present_order,
        palette=palette_map,
        edgecolor='black',
        linewidth=1.0
    )
    plt.title('Sentiment Class Distribution', fontsize=14, weight='bold', pad=12)
    plt.xlabel('Sentiment Class', fontsize=11, weight='bold')
    plt.ylabel('Count', fontsize=11, weight='bold')

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f'{int(height)}',
            (p.get_x() + p.get_width() / 2., height),
            ha='center',
            va='bottom',
            fontsize=11,
            weight='bold',
            xytext=(0, 4),
            textcoords='offset points'
        )

    plt.tight_layout()
    dist_path = os.path.join(output_dir, 'sentiment_distribution.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sentiment distribution to: {dist_path}")

    # 2. Review Length Distribution (Word & Character Lengths)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Word count histogram
    sns.histplot(
        data=df_eda,
        x='word_count',
        hue=label_col,
        hue_order=present_order,
        palette=palette_map,
        bins=35,
        kde=True,
        ax=axes[0],
        element='step'
    )
    axes[0].set_title('Review Word Count Distribution by Sentiment', fontsize=12, weight='bold')
    axes[0].set_xlabel('Number of Words', fontsize=10, weight='bold')
    axes[0].set_ylabel('Frequency', fontsize=10, weight='bold')
    axes[0].set_xlim(0, df_eda['word_count'].quantile(0.98))

    # Character count histogram
    sns.histplot(
        data=df_eda,
        x='char_count',
        hue=label_col,
        hue_order=present_order,
        palette=palette_map,
        bins=35,
        kde=True,
        ax=axes[1],
        element='step'
    )
    axes[1].set_title('Review Character Length Distribution by Sentiment', fontsize=12, weight='bold')
    axes[1].set_xlabel('Number of Characters', fontsize=10, weight='bold')
    axes[1].set_ylabel('Frequency', fontsize=10, weight='bold')
    axes[1].set_xlim(0, df_eda['char_count'].quantile(0.98))

    plt.tight_layout()
    len_path = os.path.join(output_dir, 'review_length_distribution.png')
    plt.savefig(len_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved review length distribution to: {len_path}")

    # 3. Top 15 Words Per Class
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    from src.preprocess import clean_text

    cleaned_texts = df_eda[text_col].astype(str).apply(clean_text)
    df_eda['clean_tokens'] = cleaned_texts.apply(lambda s: s.split())

    for idx, sent in enumerate(present_order):
        subset_tokens = [tok for tokens in df_eda[df_eda[label_col] == sent]['clean_tokens'] for tok in tokens]
        top15 = Counter(subset_tokens).most_common(15)

        if top15:
            words, counts = zip(*top15)
        else:
            words, counts = ['none'], [0]

        sns.barplot(
            x=list(counts),
            y=list(words),
            ax=axes[idx],
            color=palette_map.get(sent, '#4361EE'),
            edgecolor='black',
            linewidth=0.8
        )
        axes[idx].set_title(f'Top 15 Words: {sent.capitalize()}', fontsize=12, weight='bold')
        axes[idx].set_xlabel('Frequency', fontsize=10, weight='bold')
        axes[idx].set_ylabel('Tokens', fontsize=10, weight='bold')

    plt.tight_layout()
    top_words_path = os.path.join(output_dir, 'top_words_per_class.png')
    plt.savefig(top_words_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved top words chart to: {top_words_path}")

    # 4. Word Clouds per Sentiment Class
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    cmaps = {'positive': 'Greens', 'neutral': 'YlOrBr', 'negative': 'Reds'}

    for idx, sent in enumerate(present_order):
        sent_text = " ".join(df_eda[df_eda[label_col] == sent]['clean_tokens'].apply(lambda toks: " ".join(toks)))
        if not sent_text.strip():
            sent_text = f"{sent} review empty"

        wc = WordCloud(
            width=800,
            height=450,
            background_color='white',
            colormap=cmaps.get(sent, 'viridis'),
            max_words=100,
            random_state=42
        ).generate(sent_text)

        axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].axis('off')
        axes[idx].set_title(f'{sent.capitalize()} Reviews WordCloud', fontsize=14, weight='bold', pad=10)

    plt.tight_layout()
    wc_path = os.path.join(output_dir, 'wordclouds.png')
    plt.savefig(wc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved word clouds to: {wc_path}")
