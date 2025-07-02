import os
import pandas as pd
import gensim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy import spatial
import re
from collections import Counter
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

# Define the same word groups as in your extraction script
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

# Define time split
TIME_SPLIT = 1880
# Define year range for analysis
YEAR_START = 1800
YEAR_END = 1960

# ==================== UTILITY FUNCTIONS ====================

def intersection_align_gensim(m1, m2, words=None):
    """Intersect two gensim word2vec models, m1 and m2. Only the shared vocabulary between them is kept."""
    # Get the vocab for each model
    vocab_m1 = set(m1.wv.index_to_key)
    vocab_m2 = set(m2.wv.index_to_key)

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1, m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w, "count"), reverse=True)

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.key_to_index[w] for w in common_vocab]
        old_arr = m.wv.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        new_key_to_index = {}
        new_index_to_key = []
        for new_index, key in enumerate(common_vocab):
            new_key_to_index[key] = new_index
            new_index_to_key.append(key)
        m.wv.key_to_index = new_key_to_index
        m.wv.index_to_key = new_index_to_key
        
        print(len(m.wv.key_to_index), len(m.wv.vectors))
        
    return (m1, m2)

def cosine_similarity(word, model1, model2):
    """Calculate cosine similarity between embeddings of a word in two models"""
    sc = 1 - spatial.distance.cosine(model1.wv[word], model2.wv[word])
    return sc

# ==================== MAIN ANALYSIS ====================

print("=" * 80)
print("DIACHRONIC WORD2VEC ANALYSIS FOR MOOR AND INBOORLING")
print(f"SPLIT POINT: {TIME_SPLIT}")
print("=" * 80)

# Step 1: Load and preprocess data
print("\n📂 LOADING DATA...")
input_filename = 'combined_contexts_1623_2025.csv'

try:
    df = pd.read_csv(input_filename)
    print(f"✓ Successfully loaded: {input_filename}")
    print(f"Data shape: {df.shape}")
except FileNotFoundError:
    print(f"❌ Error: Could not find {input_filename}")
    exit(1)

# Check required columns
required_columns = ['context', 'year']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"❌ Error: Missing columns: {missing_columns}")
    exit(1)

print(f"Year range: {df['year'].min()}-{df['year'].max()}")

# Step 2: Normalize contexts
print("\n🔄 NORMALIZING CONTEXTS...")
df['normalized_context'] = df['context'].apply(normalize_signature_for_embeddings)
df['tokenized_context'] = df['normalized_context'].str.split()

# Process years and filter out invalid data
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year', 'tokenized_context'])
df = df[df['tokenized_context'].astype(str) != '[]']

# Filter data to only include years 1800-1960
df = df[(df['year'] >= YEAR_START) & (df['year'] <= YEAR_END)]

print(f"Data shape after preprocessing and year filtering: {df.shape}")
print(f"Year range after filtering: {df['year'].min()}-{df['year'].max()}")

# Step 3: Split data by time periods
print(f"\n📅 SPLITTING DATA BY TIME PERIODS (SPLIT: {TIME_SPLIT})...")

def assign_time_period(year):
    if year < TIME_SPLIT:
        return 'pre_1880'
    else:
        return 'post_1880'

df['time_period'] = df['year'].apply(assign_time_period)

# Show time period distribution
print("Time period distribution:")
pre_1880_data = df[df['time_period'] == 'pre_1880']
post_1880_data = df[df['time_period'] == 'post_1880']
print(f"  Pre-1880 (1800-1879): {len(pre_1880_data):,} contexts")
print(f"  Post-1880 (1880-1960): {len(post_1880_data):,} contexts")

# Step 4: Identify target words
print("\n🎯 IDENTIFYING TARGET WORDS...")

def identify_target_word(tokenized_context):
    if not isinstance(tokenized_context, list):
        return None
    
    has_moor = 'moor' in tokenized_context
    has_inboorling = 'inboorling' in tokenized_context
    
    if has_moor and has_inboorling:
        return 'both'
    elif has_moor:
        return 'moor'
    elif has_inboorling:
        return 'inboorling'
    else:
        return None

df['target_word'] = df['tokenized_context'].apply(identify_target_word)

# Filter out contexts with no target words or both words
df_filtered = df[df['target_word'].isin(['moor', 'inboorling'])].copy()
print(f"Filtered data shape: {df_filtered.shape}")

# ==================== TRAIN EMBEDDINGS FOR EACH WORD AND TIME PERIOD ====================

print("\n" + "=" * 80)
print("TRAINING EMBEDDINGS FOR EACH TARGET WORD AND TIME PERIOD")
print("=" * 80)

models = {}  # models[word][period] = model
target_words = ['moor', 'inboorling']

for target_word in target_words:
    print(f"\n{'=' * 60}")
    print(f"ANALYZING '{target_word.upper()}'")
    print(f"{'=' * 60}")
    
    models[target_word] = {}
    
    for period in ['pre_1880', 'post_1880']:
        period_label = 'Pre-1880' if period == 'pre_1880' else 'Post-1880'
        print(f"\n{'-' * 40}")
        print(f"PERIOD: {period_label}")
        print(f"{'-' * 40}")
        
        # Filter data for this word and time period
        word_period_df = df_filtered[
            (df_filtered['target_word'] == target_word) & 
            (df_filtered['time_period'] == period)
        ].copy()
        
        print(f"Data for '{target_word}' in {period_label}: {len(word_period_df):,} contexts")
        
        if len(word_period_df) < 100:
            print(f"⚠️  Warning: Only {len(word_period_df)} contexts. May be insufficient for reliable embeddings.")
            continue
        
        # Extract sentences and count tokens
        sentences = []
        token_frequency = Counter()
        
        for tokens in word_period_df['tokenized_context']:
            if isinstance(tokens, list) and len(tokens) > 0:
                sentences.append(tokens)
                token_frequency.update(tokens)
        
        print(f"Total sentences: {len(sentences):,}")
        print(f"Unique tokens: {len(token_frequency):,}")
        print(f"'{target_word}' frequency: {token_frequency.get(target_word, 0):,}")
        
        # ==================== PARAMETER GRID SEARCH ====================
        
        print(f"\n🔬 PARAMETER GRID SEARCH FOR '{target_word}' ({period_label}):")
        
        # Define parameter grid optimized for historical text
        param_grid = {
            'sg': [0, 1],  # 0 = CBOW, 1 = Skip-gram
            'epochs': [5, 10, 20],  # More training epochs
            'vector_size': [100, 200, 300],  # Smaller dimensions for limited data
            'window': [3, 5],  # Smaller context windows
            'min_count': [5, 10, 15]  # Higher minimum counts to filter noise
        }
        
        # Generate all parameter combinations
        param_combinations = list(product(
            param_grid['sg'],
            param_grid['epochs'], 
            param_grid['vector_size'],
            param_grid['window'],
            param_grid['min_count']
        ))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        best_params = None
        best_score = -1
        best_model = None
        
        for i, (sg, epochs, vector_size, window, min_count) in enumerate(param_combinations):
            # Check if target word will be included with this min_count
            target_count = token_frequency.get(target_word, 0)
            if target_count < min_count:
                continue
            
            try:
                # Train model
                test_model = gensim.models.Word2Vec(
                    sentences,
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
                    sg=sg,
                    workers=4,
                    epochs=epochs,
                    alpha=0.025,
                    min_alpha=0.0001
                )
                
                # Check if target word is in vocabulary
                if target_word not in test_model.wv.key_to_index:
                    continue
                
                # Calculate evaluation score (average similarity with top neighbors)
                try:
                    neighbors = test_model.wv.most_similar(target_word, topn=10)
                    if len(neighbors) > 0:
                        similarities = []
                        target_vector = test_model.wv[target_word]
                        for neighbor, _ in neighbors:
                            neighbor_vector = test_model.wv[neighbor]
                            similarity = 1 - cosine(target_vector, neighbor_vector)
                            similarities.append(similarity)
                        
                        if len(similarities) > 0:
                            avg_similarity = np.mean(similarities)
                            
                            # Check if this is the best model so far
                            if avg_similarity > best_score:
                                best_score = avg_similarity
                                best_params = {
                                    'sg': sg,
                                    'epochs': epochs,
                                    'vector_size': vector_size,
                                    'window': window,
                                    'min_count': min_count
                                }
                                best_model = test_model
                except:
                    continue
                    
            except Exception as e:
                continue
        
        if best_model is not None:
            print(f"🏆 BEST PARAMETERS: {best_params}")
            print(f"Best score: {best_score:.4f}")
            print(f"Vocabulary size: {len(best_model.wv.index_to_key):,}")
            
            models[target_word][period] = best_model
        else:
            print(f"❌ No valid model found")

# ==================== COSINE SIMILARITY ANALYSIS ====================

print("\n" + "=" * 80)
print("COSINE SIMILARITY ANALYSIS")
print("=" * 80)

cosine_results = {}

for target_word in target_words:
    if 'pre_1880' in models[target_word] and 'post_1880' in models[target_word]:
        print(f"\n{'=' * 50}")
        print(f"COSINE SIMILARITY FOR '{target_word.upper()}'")
        print(f"{'=' * 50}")
        
        model1 = models[target_word]['pre_1880']
        model2 = models[target_word]['post_1880']
        
        print(f"Model 1 (Pre-1880) vocabulary: {len(model1.wv.index_to_key):,} words")
        print(f"Model 2 (Post-1880) vocabulary: {len(model2.wv.index_to_key):,} words")
        
        # Check vector dimensions
        model1_dim = model1.wv.vector_size
        model2_dim = model2.wv.vector_size
        print(f"Model 1 vector dimension: {model1_dim}")
        print(f"Model 2 vector dimension: {model2_dim}")
        
        if model1_dim != model2_dim:
            print(f"⚠️  Vector dimensions don't match ({model1_dim} vs {model2_dim})")
            print("Cannot perform cosine similarity comparison with different dimensions.")
            print("Please use fixed vector_size in parameter grid or retrain models.")
            continue
        
        # Align vocabularies
        print("🔄 Aligning vocabularies...")
        model1, model2 = intersection_align_gensim(model1, model2)
        
        print(f"Aligned vocabulary size: {len(model1.wv.index_to_key):,} words")
        
        # Calculate cosine similarity for target word
        if target_word in model1.wv.index_to_key:
            target_cosine = cosine_similarity(target_word, model1, model2)
            print(f"\n🎯 Cosine similarity for '{target_word}': {target_cosine:.4f}")
            cosine_results[target_word] = target_cosine
        else:
            print(f"⚠️  '{target_word}' not in aligned vocabulary")
            cosine_results[target_word] = None
        
        # Calculate cosine similarity for all words in vocabulary
        print("\n📊 Calculating cosine similarity for all words...")
        cosine_similarity_df = pd.DataFrame([
            [w, cosine_similarity(w, model1, model2), 
             model1.wv.get_vecattr(w, "count"), 
             model2.wv.get_vecattr(w, "count")] 
            for w in model1.wv.index_to_key
        ], columns=['Word', 'Cosine_similarity', 'Frequency_t1', 'Frequency_t2'])
        
        # Save cosine similarity results
        cosine_similarity_df.to_csv(f'cosine_similarity_{target_word}.csv', index=False)
        print(f"✓ Saved cosine similarity data to 'cosine_similarity_{target_word}.csv'")
        
        # Visualize the distribution of semantic similarity scores
        plt.figure(figsize=(10, 6))
        hist = cosine_similarity_df['Cosine_similarity'].hist(bins=50, alpha=0.7)
        plt.title(f'Distribution of Cosine Similarity Scores - {target_word.title()}')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.axvline(x=cosine_similarity_df['Cosine_similarity'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {cosine_similarity_df["Cosine_similarity"].mean():.3f}')
        if target_word in model1.wv.index_to_key:
            plt.axvline(x=target_cosine, color='green', linestyle='-', linewidth=2,
                       label=f'{target_word}: {target_cosine:.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'cosine_similarity_distribution_{target_word}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Saved histogram to 'cosine_similarity_distribution_{target_word}.png'")
        
        # Statistics
        print(f"\n📈 COSINE SIMILARITY STATISTICS:")
        print(f"Mean: {cosine_similarity_df['Cosine_similarity'].mean():.4f}")
        print(f"Standard deviation: {cosine_similarity_df['Cosine_similarity'].std():.4f}")
        print(f"Min: {cosine_similarity_df['Cosine_similarity'].min():.4f}")
        print(f"Max: {cosine_similarity_df['Cosine_similarity'].max():.4f}")

# ==================== FINAL COMPARISON ====================

print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

if len(cosine_results) == 2:
    print(f"\n📊 COSINE SIMILARITY COMPARISON:")
    print(f"Moor: {cosine_results.get('moor', 'N/A')}")
    print(f"Inboorling: {cosine_results.get('inboorling', 'N/A')}")
    
    # Create comparison visualization
    valid_results = {k: v for k, v in cosine_results.items() if v is not None}
    
    if len(valid_results) > 0:
        plt.figure(figsize=(10, 6))
        words = list(valid_results.keys())
        similarities = list(valid_results.values())
        
        bars = plt.bar(words, similarities, color=['#1f77b4', '#ff7f0e'])
        plt.title('Cosine Similarity: Pre-1880 vs Post-1880 Embeddings', fontweight='bold')
        plt.ylabel('Cosine Similarity')
        
        # Add value labels on bars
        for bar, sim in zip(bars, similarities):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{sim:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig('cosine_similarity_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Saved comparison chart to 'cosine_similarity_comparison.png'")

print(f"\n📁 OUTPUT FILES:")
output_files = []
for word in target_words:
    if word in cosine_results:
        output_files.extend([
            f'cosine_similarity_{word}.csv',
            f'cosine_similarity_distribution_{word}.png'
        ])
output_files.append('cosine_similarity_comparison.png')

for f in output_files:
    print(f"   - {f}")

print(f"\n✅ ANALYSIS COMPLETE!")
print("=" * 80)