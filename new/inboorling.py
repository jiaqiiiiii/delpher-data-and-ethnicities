import os
import pandas as pd
import gensim
import numpy as np
import ast
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TEMPORAL WORD2VEC TRAINING FOR INBOORLING ANALYSIS")
print("=" * 80)

# Load processed data
processed_file = 'combined_contexts_1623_2025_processed.csv'

if not os.path.exists(processed_file):
    print(f"❌ Processed file not found: {processed_file}")
    print("Please run the data processing script first!")
    exit(1)

print(f"📂 Loading processed data from: {processed_file}")
df = pd.read_csv(processed_file)

# Convert string representations back to lists
print("🔄 Converting string columns back to lists...")
try:
    df['Tokens'] = df['Tokens'].apply(ast.literal_eval)
    df['Lemmas'] = df['Lemmas'].apply(ast.literal_eval)
    df['Lemmas_clean'] = df['Lemmas_clean'].apply(ast.literal_eval)
    df['Lemmas_nostop'] = df['Lemmas_nostop'].apply(ast.literal_eval)
    print("✅ Successfully converted columns back to lists!")
except Exception as e:
    print(f"❌ Error converting columns: {e}")
    exit(1)

# Filter for inboorling data only
print("\n🎯 Filtering for 'inboorling' data...")
df_inboorling = df[df['word_group'] == 'inboorling'].copy()
print(f"✅ Inboorling dataset: {len(df_inboorling):,} rows")

# Check year column and data distribution
print(f"\n📊 Year range: {df_inboorling['year'].min()} - {df_inboorling['year'].max()}")
print(f"Total years covered: {df_inboorling['year'].max() - df_inboorling['year'].min()}")

# Define time periods based on historical analysis
print("\n📅 DEFINING TIME PERIODS:")
print("Period 1 (1670s-1850s): Neutral usage phase")
print("Period 2 (1850s-1900): Transition phase") 
print("Period 3 (1900-1945): Colonial peak usage")
print("Period 4 (1945-1980): Decolonization period")
print("Period 5 (1980-2000s): Contemporary obsolescence")

# Define period boundaries
period_boundaries = {
    'period_1': (df_inboorling['year'].min(), 1850),
    'period_2': (1851, 1900),
    'period_3': (1901, 1945), 
    'period_4': (1946, 1980),
    'period_5': (1981, df_inboorling['year'].max())
}

print(f"\n📋 Period boundaries:")
for period, (start, end) in period_boundaries.items():
    print(f"  {period}: {start} - {end}")

# Create time period ranges and split data
print("\n🔪 SPLITTING DATA INTO TIME PERIODS:")

datasets = {}
total_check = 0

for period_name, (start_year, end_year) in period_boundaries.items():
    # Create year range for this period
    time_period_range = list(range(start_year, end_year + 1))
    
    # Filter data for this period
    period_data = df_inboorling[df_inboorling['year'].isin(time_period_range)]
    datasets[period_name] = period_data
    
    total_check += len(period_data)
    
    print(f"  {period_name} ({start_year}-{end_year}): {len(period_data):,} rows")
    if len(period_data) > 0:
        print(f"    Year range in data: {period_data['year'].min()} - {period_data['year'].max()}")

# Verify split is complete
print(f"\n✅ Data split verification:")
print(f"  Original data: {len(df_inboorling):,} rows")
print(f"  Sum of periods: {total_check:,} rows")
print(f"  Match: {len(df_inboorling) == total_check}")

# Best parameters for inboorling (from previous analysis)
best_params = {
    'sg': 1,           # Skip-gram
    'vector_size': 300,
    'window': 10,
    'min_count': 0     # Include all words
}

print(f"\n🎯 TRAINING WORD2VEC MODELS WITH BEST PARAMETERS:")
print(f"  Architecture: Skip-gram (sg=1)")
print(f"  Vector size: {best_params['vector_size']}")
print(f"  Window size: {best_params['window']}")
print(f"  Min count: {best_params['min_count']}")

# Function to print vocabulary sample
def print_vocab(model, n=10):
    """Print first n words from model vocabulary"""
    vocab_words = list(model.wv.index_to_key)
    print(f"  Vocabulary sample ({n} words): {vocab_words[:n]}")

# Train models for each period
models = {}
vocab_sizes = {}

print(f"\n🔄 TRAINING MODELS:")

for period_name, period_data in datasets.items():
    if len(period_data) == 0:
        print(f"\n❌ {period_name}: No data available - skipping")
        continue
    
    print(f"\n🎯 Training model for {period_name}...")
    print(f"  Data size: {len(period_data):,} contexts")
    
    try:
        # Train Word2Vec model
        model = gensim.models.Word2Vec(
            period_data['Lemmas_clean'], 
            min_count=best_params['min_count'],
            vector_size=best_params['vector_size'],
            window=best_params['window'],
            sg=best_params['sg']
        )
        
        models[period_name] = model
        vocab_size = len(list(model.wv.index_to_key))
        vocab_sizes[period_name] = vocab_size
        
        print(f"  ✅ Model trained successfully!")
        print(f"  📊 Vocabulary size: {vocab_size:,} words")
        
        # Print vocabulary sample
        print_vocab(model, 10)
        
        # Save model
        model_filename = f'word2vec_inboorling_{period_name}_sg_w10_f0_300.model'
        model.save(model_filename)
        print(f"  💾 Model saved as: {model_filename}")
        
    except Exception as e:
        print(f"  ❌ Error training model for {period_name}: {e}")
        continue

print(f"\n📊 VOCABULARY SIZES SUMMARY:")
for period_name, vocab_size in vocab_sizes.items():
    start, end = period_boundaries[period_name]
    print(f"  {period_name} ({start}-{end}): {vocab_size:,} words")

# Calculate vocabulary intersections
print(f"\n🔍 VOCABULARY INTERSECTIONS:")

if len(models) >= 2:
    period_names = list(models.keys())
    vocabs = {}
    
    # Create vocabulary sets
    for period_name in period_names:
        vocabs[period_name] = set(list(models[period_name].wv.index_to_key))
    
    # Calculate pairwise intersections
    print(f"\n📋 Pairwise vocabulary intersections:")
    for i, period1 in enumerate(period_names):
        for j, period2 in enumerate(period_names):
            if i < j:  # Only calculate each pair once
                intersection_size = len(vocabs[period1].intersection(vocabs[period2]))
                vocab1_size = len(vocabs[period1])
                vocab2_size = len(vocabs[period2])
                
                # Calculate Jaccard similarity
                union_size = len(vocabs[period1].union(vocabs[period2]))
                jaccard = intersection_size / union_size if union_size > 0 else 0
                
                print(f"  {period1} ∩ {period2}:")
                print(f"    Intersection: {intersection_size:,} words")
                print(f"    Jaccard similarity: {jaccard:.4f}")
                print(f"    Coverage: {intersection_size/vocab1_size:.2%} of {period1}, {intersection_size/vocab2_size:.2%} of {period2}")
    
    # Calculate intersection of ALL periods
    if len(models) > 2:
        all_intersection = vocabs[period_names[0]]
        for period_name in period_names[1:]:
            all_intersection = all_intersection.intersection(vocabs[period_name])
        
        print(f"\n🎯 COMMON VOCABULARY ACROSS ALL PERIODS:")
        print(f"  Words appearing in all {len(models)} periods: {len(all_intersection):,}")
        if len(all_intersection) > 0:
            common_words = sorted(list(all_intersection))[:20]  # Show first 20
            print(f"  Sample common words: {common_words}")

else:
    print("❌ Need at least 2 models to calculate intersections")

# Check if 'inboorling' appears in each model
print(f"\n🎯 TARGET WORD 'inboorling' IN VOCABULARIES:")
target_word = 'inboorling'

for period_name, model in models.items():
    if target_word in model.wv.key_to_index:
        print(f"  ✅ {period_name}: '{target_word}' found in vocabulary")
        
        # Get most similar words
        try:
            similar_words = model.wv.most_similar(target_word, topn=10)
            similar_list = [word for word, score in similar_words]
            print(f"    Top 10 similar: {similar_list}")
        except Exception as e:
            print(f"    ❌ Error getting similar words: {e}")
    else:
        print(f"  ❌ {period_name}: '{target_word}' NOT found in vocabulary")

print(f"\n✅ TEMPORAL ANALYSIS COMPLETE!")
print(f"📁 Saved models:")
for period_name in models.keys():
    model_filename = f'word2vec_inboorling_{period_name}_sg_w10_f0_300.model'
    print(f"  - {model_filename}")

print("=" * 80)