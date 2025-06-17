import os
import pandas as pd
import gensim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import itertools
import re
from collections import Counter
import seaborn as sns

# Define the same word groups as in your original script
word_groups = {
    "moor": {
        "base_word": "moor",
        "variations": ["moor", "moren", "moorin", "mooren", "moore", "moors", "moorse", "moorsche"]
    },
    "inboorling": {
        "base_word": "inboorling", 
        "variations": ["inboorling", "inboorlingen", "inboorlingh", "inboorlinghen"]
    }
}

# Create mapping from variations to base words
variation_to_base = {}
for group_name, group_data in word_groups.items():
    for variation in group_data["variations"]:
        variation_to_base[variation.lower()] = group_data["base_word"]

def normalize_signature_for_embeddings(signature_text):
    """Replace all spelling variations with their base words in the signature text"""
    if not signature_text or pd.isna(signature_text):
        return ""
    
    words = signature_text.split()
    normalized_words = []
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word in variation_to_base:
            normalized_words.append(variation_to_base[clean_word])
        else:
            normalized_words.append(word)
    
    return ' '.join(normalized_words)

# Load data and preprocess
print("Loading and preprocessing data...")
df = pd.read_csv('1700_unique_contexts_by_word.csv', sep=';')
print("Columns in df:", repr(df.columns.tolist()))

# Normalize signatures to replace variations with base words
df['normalized_signature'] = df['signature'].apply(normalize_signature_for_embeddings)
df['tokenized_signature'] = df['normalized_signature'].str.split()

df['year'] = pd.to_datetime(df['year'], format="%Y", errors='coerce').dt.year
df = df.dropna(subset=['year', 'tokenized_signature'])
df['decade'] = (df['year'] // 10) * 10

print(f"Data shape after preprocessing: {df.shape}")
print(f"Decades found: {sorted(df['decade'].unique())}")

# Check word frequencies after normalization
all_tokens = []
for _, grp in df.groupby('decade'):
    for sentence in grp['tokenized_signature'].tolist():
        all_tokens.extend(sentence)

word_counts = Counter(all_tokens)
target_words = ["moor", "inboorling"]

print("\nWord frequencies after normalization:")
for word in target_words:
    count = word_counts.get(word, 0)
    print(f"  '{word}': {count} occurrences")

# Define parameter grid to tune
param_grid = {
    'vector_size': [50, 100, 200, 300],
    'window': [3, 5, 7, 10],
    'min_count': [1, 2, 3, 4, 5],
    'sg': [0, 1]  # 0 = CBOW, 1 = Skip-gram
}

def calculate_semantic_change_metrics(sim_df):
    """Calculate various metrics to evaluate semantic change detection"""
    metrics = {}
    
    for word in target_words:
        word_data = sim_df[sim_df['word'] == word]
        if len(word_data) == 0:
            continue
            
        # 1. Variance in mean similarities (higher = more change)
        variance_mean_sim = np.var(word_data['mean_sim'])
        
        # 2. Mean of standard deviations (consistency across comparisons)
        mean_std = np.mean(word_data['std_sim'])
        
        # 3. Range of similarities (max - min)
        sim_range = np.max(word_data['mean_sim']) - np.min(word_data['mean_sim'])
        
        # 4. Trend strength (absolute correlation with time)
        decades = word_data['decade'].values
        similarities = word_data['mean_sim'].values
        if len(decades) > 2:
            correlation = np.abs(np.corrcoef(decades, similarities)[0, 1])
        else:
            correlation = 0
        
        # 5. Number of decades with data
        n_decades = len(word_data)
        
        metrics[word] = {
            'variance_mean_sim': variance_mean_sim,
            'mean_std': mean_std,
            'sim_range': sim_range,
            'trend_strength': correlation,
            'n_decades': n_decades,
            'coverage': n_decades / len(df['decade'].unique())  # Proportion of decades covered
        }
    
    return metrics

def train_and_evaluate(params):
    print(f"Training with params: {params}")
    
    # Train Word2Vec per decade
    models = {}
    vocab_info = {}
    
    for decade, grp in df.groupby('decade'):
        sentences = grp['tokenized_signature'].tolist()
        
        # Skip if no sentences
        if not sentences:
            continue
            
        model = gensim.models.Word2Vec(
            sentences,
            vector_size=params['vector_size'],
            window=params['window'],
            min_count=params['min_count'],
            sg=params['sg'],
            workers=4,
            epochs=10  # Add more epochs for better training
        )
        models[decade] = model
        
        # Track vocabulary info
        vocab_info[decade] = {
            'vocab_size': len(model.wv.key_to_index),
            'target_words_present': [w for w in target_words if w in model.wv.key_to_index],
            'n_sentences': len(sentences)
        }

    # Compute cosine similarities for target words across decades
    records = []
    for word in target_words:
        decades_with_word = [d for d, m in models.items() if word in m.wv.key_to_index]
        
        for dec in decades_with_word:
            vec = models[dec].wv[word]
            sims = []
            
            for dec2 in decades_with_word:
                if dec2 != dec:
                    similarity = 1 - cosine(vec, models[dec2].wv[word])
                    sims.append(similarity)
            
            if sims:
                records.append({
                    "word": word,
                    "decade": dec,
                    "mean_sim": np.mean(sims),
                    "std_sim": np.std(sims),
                    "n_comparisons": len(sims)
                })

    sim_df = pd.DataFrame(records)
    
    # Calculate evaluation metrics
    metrics = calculate_semantic_change_metrics(sim_df)
    
    return sim_df, params, metrics, vocab_info

# Run parameter tuning
print("Starting parameter tuning...")
results = []
param_performance = []

total_combinations = len(list(itertools.product(*param_grid.values())))
print(f"Testing {total_combinations} parameter combinations...")

for i, param_combo in enumerate(itertools.product(*param_grid.values())):
    params = dict(zip(param_grid.keys(), param_combo))
    
    print(f"\nProgress: {i+1}/{total_combinations}")
    
    try:
        sim_df, used_params, metrics, vocab_info = train_and_evaluate(params)
        
        # Add parameter info to results
        for col, val in used_params.items():
            sim_df[col] = val
        
        results.append(sim_df)
        
        # Store parameter performance
        param_performance.append({
            **used_params,
            **{f"{word}_{metric}": value for word, word_metrics in metrics.items() 
               for metric, value in word_metrics.items()},
            'total_records': len(sim_df),
            'avg_coverage': np.mean([metrics[word]['coverage'] for word in metrics.keys()])
        })
        
        print(f"  Results: {len(sim_df)} records, avg coverage: {np.mean([metrics[word]['coverage'] for word in metrics.keys()]):.2f}")
        
    except Exception as e:
        print(f"  Error with params {params}: {e}")
        continue

# Concatenate all results
if results:
    all_results = pd.concat(results, ignore_index=True)
    param_df = pd.DataFrame(param_performance)
    
    # Save results
    all_results.to_csv('word2vec_semantic_change_results_normalized.csv', index=False)
    param_df.to_csv('parameter_performance_analysis.csv', index=False)
    
    print(f"\nParameter tuning complete! Found {len(param_df)} successful parameter combinations.")
    
    # Find best parameters based on different criteria
    print("\n=== BEST PARAMETER ANALYSIS ===")
    
    # 1. Best for capturing semantic change (high variance for inboorling, low for moor)
    if 'inboorling_variance_mean_sim' in param_df.columns and 'moor_variance_mean_sim' in param_df.columns:
        # We want high variance for inboorling (captures change) and reasonable coverage
        param_df['semantic_change_score'] = (
            param_df['inboorling_variance_mean_sim'] * 2 +  # Weight inboorling change more
            param_df['avg_coverage'] * 0.5 +  # Ensure good coverage
            (1 - param_df['moor_variance_mean_sim'])  # Prefer stable moor (low variance)
        )
        
        best_semantic_idx = param_df['semantic_change_score'].idxmax()
        best_semantic_params = param_df.loc[best_semantic_idx]
        
        print(f"Best for semantic change detection:")
        print(f"  Vector size: {best_semantic_params['vector_size']}")
        print(f"  Window: {best_semantic_params['window']}")
        print(f"  Min count: {best_semantic_params['min_count']}")
        print(f"  Algorithm: {'Skip-gram' if best_semantic_params['sg'] == 1 else 'CBOW'}")
        print(f"  Inboorling variance: {best_semantic_params['inboorling_variance_mean_sim']:.4f}")
        print(f"  Moor variance: {best_semantic_params['moor_variance_mean_sim']:.4f}")
        print(f"  Coverage: {best_semantic_params['avg_coverage']:.2f}")
    
    # 2. Best overall coverage
    best_coverage_idx = param_df['avg_coverage'].idxmax()
    best_coverage_params = param_df.loc[best_coverage_idx]
    
    print(f"\nBest for coverage:")
    print(f"  Vector size: {best_coverage_params['vector_size']}")
    print(f"  Window: {best_coverage_params['window']}")
    print(f"  Min count: {best_coverage_params['min_count']}")
    print(f"  Algorithm: {'Skip-gram' if best_coverage_params['sg'] == 1 else 'CBOW'}")
    print(f"  Coverage: {best_coverage_params['avg_coverage']:.2f}")
    
    # 3. Create visualization for best semantic change parameters
    if 'semantic_change_score' in param_df.columns:
        best_params = {
            'vector_size': int(best_semantic_params['vector_size']),
            'window': int(best_semantic_params['window']),
            'min_count': int(best_semantic_params['min_count']),
            'sg': int(best_semantic_params['sg'])
        }
        
        subset = all_results[
            (all_results['vector_size'] == best_params['vector_size']) &
            (all_results['window'] == best_params['window']) &
            (all_results['min_count'] == best_params['min_count']) &
            (all_results['sg'] == best_params['sg'])
        ]
        
        if not subset.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            for word, grp in subset.groupby("word"):
                ax.errorbar(grp['decade'], grp['mean_sim'], yerr=grp['std_sim'],
                           label=f"{word} (n={len(grp)})", capsize=3, marker='o', linewidth=2)
            
            ax.set_xlabel("Decade")
            ax.set_ylabel("Mean Cosine Similarity to other decades")
            ax.set_title(f"Semantic Evolution - Best Parameters\n{best_params}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("best_semantic_change_plot_normalized.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nBest parameters visualization saved to 'best_semantic_change_plot_normalized.png'")
    
    # 4. Parameter sensitivity analysis
    print(f"\n=== PARAMETER SENSITIVITY ===")
    for param in ['vector_size', 'window', 'min_count', 'sg']:
        if f'inboorling_variance_mean_sim' in param_df.columns:
            correlation = param_df[param].corr(param_df['inboorling_variance_mean_sim'])
            print(f"{param} correlation with inboorling semantic change: {correlation:.3f}")
    
else:
    print("No successful parameter combinations found!")

print("Analysis complete!")
