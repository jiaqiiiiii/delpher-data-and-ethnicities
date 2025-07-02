import pandas as pd
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec

# Load the aligned models
model1 = Word2Vec.load('aligned_model1.model')
model2 = Word2Vec.load('aligned_model2.model')
model3 = Word2Vec.load('aligned_model3.model')
model4 = Word2Vec.load('aligned_model4.model')
model5 = Word2Vec.load('aligned_model5.model')

# Load the word variations data
print("Loading word variations from CSV...")
word_variations_df = pd.read_csv('combined_contexts_1623_2025_processed.csv')

# Get all variations of inboorling with better handling
print("\nExtracting variations of 'inboorling'...")
inboorling_df = word_variations_df[word_variations_df['word_group'] == 'inboorling']
print(f"Total rows with word_group 'inboorling': {len(inboorling_df)}")

# Get variations with cleaning
inboorling_variations_raw = inboorling_df['matched_variation'].dropna()
# Strip whitespace and convert to lowercase for consistency
inboorling_variations = [str(v).strip().lower() for v in inboorling_variations_raw.unique()]
inboorling_variations = list(set(inboorling_variations))  # Remove any duplicates

print(f"\nFound {len(inboorling_variations)} unique variations of 'inboorling':")
for i, var in enumerate(inboorling_variations, 1):
    print(f"  {i}. '{var}'")

# Also check if the variations exist in the models
print("\nChecking which variations exist in each model:")
models = [model1, model2, model3, model4, model5]
model_names = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5']

for model, name in zip(models, model_names):
    found_in_model = []
    for var in inboorling_variations:
        if var in model.wv:
            found_in_model.append(var)
    print(f"{name}: {len(found_in_model)} variations found - {found_in_model}")

def get_aggregated_embedding(word_group, variations, model):
    """
    Get aggregated embedding for a word group by averaging embeddings of all its variations.
    Returns the mean embedding vector or None if no variations found in vocabulary.
    """
    embeddings = []
    found_variations = []
    
    for variation in variations:
        # Try both the original case and lowercase
        for var_case in [variation, variation.lower(), variation.upper(), variation.capitalize()]:
            try:
                if var_case in model.wv:
                    embeddings.append(model.wv[var_case])
                    found_variations.append(var_case)
                    break  # Found it, no need to try other cases
            except KeyError:
                continue
    
    if embeddings:
        # Return average embedding
        return np.mean(embeddings, axis=0), found_variations
    else:
        return None, []

def get_aggregated_frequency(variations, model):
    """Get total frequency for all variations of a word."""
    total_freq = 0
    for variation in variations:
        # Try both the original case and lowercase
        for var_case in [variation, variation.lower(), variation.upper(), variation.capitalize()]:
            try:
                if var_case in model.wv:
                    total_freq += model.wv.get_vecattr(var_case, "count")
                    break  # Found it, count it once
            except KeyError:
                continue
    return total_freq

def cosine_similarity_aggregated(word_group, variations, model_a, model_b):
    """
    Calculate cosine similarity between aggregated embeddings of a word group in two models.
    """
    embedding_a, found_a = get_aggregated_embedding(word_group, variations, model_a)
    embedding_b, found_b = get_aggregated_embedding(word_group, variations, model_b)
    
    if embedding_a is not None and embedding_b is not None:
        similarity = 1 - spatial.distance.cosine(embedding_a, embedding_b)
        return similarity, found_a, found_b
    else:
        return np.nan, [], []

def cosine_similarity(word, model_a, model_b):
    """
    Calculate cosine similarity between embeddings of a word in two models.
    Returns 1 - cosine_distance (so higher values = more similar)
    """
    try:
        similarity = 1 - spatial.distance.cosine(model_a.wv[word], model_b.wv[word])
        return similarity
    except KeyError:
        # Word not in vocabulary
        return np.nan

# Create cosine similarity dataframes for all pairwise comparisons

# Period 1 vs Period 2
print("\nCalculating cosine similarities: Period 1 vs Period 2...")
cosine_similarity_df_1_2 = pd.DataFrame([
    [w, 
     cosine_similarity(w, model1, model2), 
     model1.wv.get_vecattr(w, "count"), 
     model2.wv.get_vecattr(w, "count")] 
    for w in model1.wv.index_to_key
], columns=['Word', 'Cosine_similarity', 'Frequency_t1', 'Frequency_t2'])

# Period 1 vs Period 3
print("Calculating cosine similarities: Period 1 vs Period 3...")
cosine_similarity_df_1_3 = pd.DataFrame([
    [w, 
     cosine_similarity(w, model1, model3), 
     model1.wv.get_vecattr(w, "count"), 
     model3.wv.get_vecattr(w, "count")] 
    for w in model1.wv.index_to_key
], columns=['Word', 'Cosine_similarity', 'Frequency_t1', 'Frequency_t3'])

# Period 1 vs Period 4
print("Calculating cosine similarities: Period 1 vs Period 4...")
cosine_similarity_df_1_4 = pd.DataFrame([
    [w, 
     cosine_similarity(w, model1, model4), 
     model1.wv.get_vecattr(w, "count"), 
     model4.wv.get_vecattr(w, "count")] 
    for w in model1.wv.index_to_key
], columns=['Word', 'Cosine_similarity', 'Frequency_t1', 'Frequency_t4'])

# Period 1 vs Period 5
print("Calculating cosine similarities: Period 1 vs Period 5...")
cosine_similarity_df_1_5 = pd.DataFrame([
    [w, 
     cosine_similarity(w, model1, model5), 
     model1.wv.get_vecattr(w, "count"), 
     model5.wv.get_vecattr(w, "count")] 
    for w in model1.wv.index_to_key
], columns=['Word', 'Cosine_similarity', 'Frequency_t1', 'Frequency_t5'])

# Process each dataframe: add total frequency, sort by semantic change
def process_similarity_df(df, freq_col1, freq_col2):
    """Process similarity dataframe: add total frequency and sort by cosine similarity"""
    df[f"Total_Frequency"] = df[freq_col1] + df[freq_col2]
    df_sorted = df.sort_values(by='Cosine_similarity', ascending=True)
    return df_sorted

print("\nProcessing dataframes...")
cosine_similarity_df_1_2_sorted = process_similarity_df(cosine_similarity_df_1_2, 'Frequency_t1', 'Frequency_t2')
cosine_similarity_df_1_3_sorted = process_similarity_df(cosine_similarity_df_1_3, 'Frequency_t1', 'Frequency_t3')
cosine_similarity_df_1_4_sorted = process_similarity_df(cosine_similarity_df_1_4, 'Frequency_t1', 'Frequency_t4')
cosine_similarity_df_1_5_sorted = process_similarity_df(cosine_similarity_df_1_5, 'Frequency_t1', 'Frequency_t5')

# ADDITION 1: Create a big dataframe with all comparisons to t1
print("\nCreating combined dataframe with all comparisons to Period 1...")
all_comparisons_df = pd.DataFrame()

# Start with the word list from model1
all_comparisons_df['Word'] = cosine_similarity_df_1_2_sorted['Word']
all_comparisons_df['Frequency_t1'] = cosine_similarity_df_1_2_sorted['Frequency_t1']

# Add cosine similarities for each comparison
all_comparisons_df['Cosine_similarity_1_2'] = cosine_similarity_df_1_2_sorted['Cosine_similarity']
all_comparisons_df['Cosine_similarity_1_3'] = cosine_similarity_df_1_3_sorted['Cosine_similarity']
all_comparisons_df['Cosine_similarity_1_4'] = cosine_similarity_df_1_4_sorted['Cosine_similarity']
all_comparisons_df['Cosine_similarity_1_5'] = cosine_similarity_df_1_5_sorted['Cosine_similarity']

# Add frequencies from other periods
all_comparisons_df['Frequency_t2'] = cosine_similarity_df_1_2_sorted['Frequency_t2']
all_comparisons_df['Frequency_t3'] = cosine_similarity_df_1_3_sorted['Frequency_t3']
all_comparisons_df['Frequency_t4'] = cosine_similarity_df_1_4_sorted['Frequency_t4']
all_comparisons_df['Frequency_t5'] = cosine_similarity_df_1_5_sorted['Frequency_t5']

# Calculate average semantic change across all periods
all_comparisons_df['Average_cosine_similarity'] = all_comparisons_df[['Cosine_similarity_1_2', 
                                                                       'Cosine_similarity_1_3', 
                                                                       'Cosine_similarity_1_4', 
                                                                       'Cosine_similarity_1_5']].mean(axis=1)

# Sort by average semantic change
all_comparisons_df_sorted = all_comparisons_df.sort_values(by='Average_cosine_similarity', ascending=True)

# Save the combined dataframe
all_comparisons_df_sorted.to_csv('all_comparisons_to_period1.csv', index=False)
print("Saved all comparisons to 'all_comparisons_to_period1.csv'")

# ADDITION 2: Extract inboorling changes over time periods using aggregated embeddings
print("\nExtracting 'inboorling' semantic changes over time using aggregated embeddings...")

# Calculate aggregated similarities for inboorling
inboorling_aggregated_data = []

# Period 1 vs 2
sim_1_2, found_1_2_a, found_1_2_b = cosine_similarity_aggregated('inboorling', inboorling_variations, model1, model2)
freq_1_2_a = get_aggregated_frequency(inboorling_variations, model1)
freq_1_2_b = get_aggregated_frequency(inboorling_variations, model2)

# Period 1 vs 3
sim_1_3, found_1_3_a, found_1_3_b = cosine_similarity_aggregated('inboorling', inboorling_variations, model1, model3)
freq_1_3_a = get_aggregated_frequency(inboorling_variations, model1)
freq_1_3_b = get_aggregated_frequency(inboorling_variations, model3)

# Period 1 vs 4
sim_1_4, found_1_4_a, found_1_4_b = cosine_similarity_aggregated('inboorling', inboorling_variations, model1, model4)
freq_1_4_a = get_aggregated_frequency(inboorling_variations, model1)
freq_1_4_b = get_aggregated_frequency(inboorling_variations, model4)

# Period 1 vs 5
sim_1_5, found_1_5_a, found_1_5_b = cosine_similarity_aggregated('inboorling', inboorling_variations, model1, model5)
freq_1_5_a = get_aggregated_frequency(inboorling_variations, model1)
freq_1_5_b = get_aggregated_frequency(inboorling_variations, model5)

# Create summary dataframe
inboorling_summary = pd.DataFrame({
    'Comparison': ['Period 1 vs 2', 'Period 1 vs 3', 'Period 1 vs 4', 'Period 1 vs 5'],
    'Cosine_similarity': [sim_1_2, sim_1_3, sim_1_4, sim_1_5],
    'Frequency_t1': [freq_1_2_a, freq_1_3_a, freq_1_4_a, freq_1_5_a],
    'Frequency_other_period': [freq_1_2_b, freq_1_3_b, freq_1_4_b, freq_1_5_b],
    'Variations_found_t1': [', '.join(found_1_2_a), ', '.join(found_1_3_a), ', '.join(found_1_4_a), ', '.join(found_1_5_a)],
    'Variations_found_other': [', '.join(found_1_2_b), ', '.join(found_1_3_b), ', '.join(found_1_4_b), ', '.join(found_1_5_b)]
})

inboorling_summary.to_csv('inboorling_semantic_change_aggregated.csv', index=False)
print("Saved 'inboorling' semantic changes to 'inboorling_semantic_change_aggregated.csv'")

# ADDITION 3: Get neighbors of 'inboorling' variations in all five time periods
print("\nFinding neighbors of 'inboorling' (all variations) in all time periods...")
with open('inboorling_neighbors_aggregated.txt', 'w', encoding='utf-8') as f:
    f.write("NEIGHBORS OF 'INBOORLING' (ALL VARIATIONS) ACROSS TIME PERIODS\n")
    f.write("=" * 50 + "\n\n")
    
    models = [model1, model2, model3, model4, model5]
    period_names = ['Period 1', 'Period 2', 'Period 3', 'Period 4', 'Period 5']
    
    for model, period_name in zip(models, period_names):
        f.write(f"{period_name}:\n")
        f.write("-" * 30 + "\n")
        
        # Get aggregated embedding for this period
        aggregated_embedding, found_variations = get_aggregated_embedding('inboorling', inboorling_variations, model)
        
        if aggregated_embedding is not None:
            f.write(f"Variations found: {', '.join(found_variations)}\n")
            f.write(f"Total frequency: {get_aggregated_frequency(inboorling_variations, model)}\n\n")
            
            # Find most similar words to the aggregated embedding
            similarities = []
            for word in model.wv.index_to_key[:1000]:  # Check top 1000 most frequent words
                try:
                    sim = 1 - spatial.distance.cosine(aggregated_embedding, model.wv[word])
                    similarities.append((word, sim))
                except:
                    continue
            
            # Sort and get top 10
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_neighbors = similarities[:15]  # Get more to filter out variations
            
            # Filter out the variations themselves
            filtered_neighbors = [(w, s) for w, s in top_neighbors if w not in inboorling_variations][:10]
            
            for i, (word, similarity) in enumerate(filtered_neighbors, 1):
                f.write(f"{i}. {word} (similarity: {similarity:.4f})\n")
        else:
            f.write("No variations of 'inboorling' found in this period's vocabulary\n")
        
        f.write("\n")

print("Saved neighbors of 'inboorling' to 'inboorling_neighbors_aggregated.txt'")

# Visualize distributions
print("\nCreating visualizations...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

datasets = [
    (cosine_similarity_df_1_2_sorted, "Period 1 vs 2"),
    (cosine_similarity_df_1_3_sorted, "Period 1 vs 3"), 
    (cosine_similarity_df_1_4_sorted, "Period 1 vs 4"),
    (cosine_similarity_df_1_5_sorted, "Period 1 vs 5"),
]

for i, (df, title) in enumerate(datasets):
    df['Cosine_similarity'].hist(bins=50, ax=axes[i], alpha=0.7)
    axes[i].set_title(f'Cosine Similarity Distribution\n{title}')
    axes[i].set_xlabel('Cosine Similarity')
    axes[i].set_ylabel('Frequency')
    
# Remove empty subplot
axes[7].remove()

plt.tight_layout()
plt.show()

# Display summary statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)

for df, title in datasets:
    print(f"\n{title}:")
    print(f"  Mean cosine similarity: {df['Cosine_similarity'].mean():.4f}")
    print(f"  Std cosine similarity: {df['Cosine_similarity'].std():.4f}")
    print(f"  Min cosine similarity: {df['Cosine_similarity'].min():.4f}")
    print(f"  Max cosine similarity: {df['Cosine_similarity'].max():.4f}")

# Look up specific word "inboorling" across all comparisons (individual variations)
print("\n" + "="*50)
print("SEMANTIC CHANGE FOR 'inboorling' VARIATIONS ACROSS TIME PERIODS")
print("="*50)

for variation in inboorling_variations:
    print(f"\nVariation: '{variation}'")
    print("-" * 30)
    for df, title in datasets:
        variation_row = df[df['Word'] == variation]
        if not variation_row.empty:
            print(f"{title}: Cosine similarity = {variation_row['Cosine_similarity'].values[0]:.4f}")

# Display top 10 most changed words for each comparison
print("\n" + "="*50)
print("TOP 10 MOST SEMANTICALLY CHANGED WORDS")
print("="*50)

for df, title in datasets:
    print(f"\n{title} - Top 10 most changed words:")
    print(df.head(10)[['Word', 'Cosine_similarity', 'Total_Frequency']].to_string(index=False))

# Save all dataframes to CSV
print("\nSaving results to CSV files...")
cosine_similarity_df_1_2_sorted.to_csv('cosine_similarity_period_1_vs_2.csv', index=False)
cosine_similarity_df_1_3_sorted.to_csv('cosine_similarity_period_1_vs_3.csv', index=False)
cosine_similarity_df_1_4_sorted.to_csv('cosine_similarity_period_1_vs_4.csv', index=False)
cosine_similarity_df_1_5_sorted.to_csv('cosine_similarity_period_1_vs_5.csv', index=False)

# ADDITION 4: Plot cosine similarity changes over time for aggregated inboorling
print("\nPlotting temporal semantic changes for 'inboorling' (aggregated)...")

# Create a dataframe for temporal analysis
temporal_df = pd.DataFrame({
    'timeslice': [1, 2, 3, 4, 5],
    'inboorling': [1.0,  # Period 1 vs itself is always 1.0
                   inboorling_summary['Cosine_similarity'].values[0] if not inboorling_summary.empty else np.nan,
                   inboorling_summary['Cosine_similarity'].values[1] if not inboorling_summary.empty else np.nan,
                   inboorling_summary['Cosine_similarity'].values[2] if not inboorling_summary.empty else np.nan,
                   inboorling_summary['Cosine_similarity'].values[3] if not inboorling_summary.empty else np.nan]
})

# Plot the temporal changes
plt.figure(figsize=(10, 6))
plt.plot(temporal_df['timeslice'], temporal_df['inboorling'], label='inboorling (aggregated)', marker='o', linewidth=2)
plt.ylim([0.1, 1.1])
plt.xlabel('Time Period')
plt.ylabel('Cosine Similarity (compared to Period 1)')
plt.title('Semantic Change of "inboorling" (All Variations) Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('inboorling_temporal_change_aggregated.png', dpi=300, bbox_inches='tight')
plt.show()

# ADDITION 5: Detect changepoints using ruptures
print("\nDetecting changepoints in semantic trajectory of 'inboorling' (aggregated)...")

try:
    import ruptures as rpt
    
    # Define parameters
    costfunction = 'l1'  # Try also 'l2' and 'rbf'
    jump = 1
    pen = 0.5
    
    # Prepare data for changepoint detection
    word = 'inboorling'
    slices = temporal_df['timeslice']
    ts = temporal_df[word]
    y = np.array(ts.tolist())
    
    # Run Pelt algorithm
    model = rpt.Pelt(model=costfunction, jump=jump)
    model.fit(y)
    breaks = model.predict(pen=pen)
    print(f"Detected changepoints: {breaks}")
    
    # Format changepoints
    breaksstr = ''
    for b in breaks:
        if b == len(slices):
            continue
        else:
            breaksstr += str(slices.iloc[b-1])
            breaksstr += ' '
    
    # Create changepoints dataframe
    changepointsdict = {'words': [word + '_aggregated'], 'changepoints': [breaksstr.strip()]}
    changepointsdf = pd.DataFrame.from_dict(changepointsdict)
    
    # Save changepoints
    changepointsdf.to_csv(f'changepoints-{word}-aggregated-{costfunction}-jump{str(jump)}-pen{str(pen)}.csv', index=False)
    print(f"Saved changepoints to 'changepoints-{word}-aggregated-{costfunction}-jump{str(jump)}-pen{str(pen)}.csv'")
    
    # Visualize changepoints on the plot
    plt.figure(figsize=(10, 6))
    plt.plot(temporal_df['timeslice'], temporal_df['inboorling'], label='inboorling (aggregated)', marker='o', linewidth=2)
    
    # Add vertical lines for changepoints
    for b in breaks[:-1]:  # Exclude the last break which is the end of the series
        plt.axvline(x=slices.iloc[b-1], color='red', linestyle='--', alpha=0.7, label=f'Changepoint at period {slices.iloc[b-1]}')
    
    plt.ylim([0.1, 1.1])
    plt.xlabel('Time Period')
    plt.ylabel('Cosine Similarity (compared to Period 1)')
    plt.title(f'Semantic Change of "inboorling" (Aggregated) with Changepoints\n(cost={costfunction}, jump={jump}, penalty={pen})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'inboorling_changepoints_aggregated_{costfunction}_jump{jump}_pen{pen}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Try different parameter combinations
    print("\nTrying different changepoint detection parameters...")
    parameter_combinations = [
        ('l1', 1, 0.5),
        ('l1', 1, 1.0),
        ('l2', 1, 0.5),
        ('l2', 1, 1.0),
        ('rbf', 1, 0.5),
        ('rbf', 1, 1.0)
    ]
    
    all_changepoints = []
    for cost, j, p in parameter_combinations:
        model = rpt.Pelt(model=cost, jump=j)
        model.fit(y)
        breaks = model.predict(pen=p)
        
        breaksstr = ''
        for b in breaks:
            if b == len(slices):
                continue
            else:
                breaksstr += str(slices.iloc[b-1])
                breaksstr += ' '
        
        all_changepoints.append({
            'cost_function': cost,
            'jump': j,
            'penalty': p,
            'changepoints': breaksstr.strip()
        })
    
    # Save all changepoint results
    all_changepoints_df = pd.DataFrame(all_changepoints)
    all_changepoints_df.to_csv('inboorling_changepoints_aggregated_all_parameters.csv', index=False)
    print("Saved all changepoint detection results to 'inboorling_changepoints_aggregated_all_parameters.csv'")
    
except ImportError:
    print("\nWARNING: 'ruptures' library not installed. Skipping changepoint detection.")
    print("To install, run: pip install ruptures")

print("\nAnalysis complete! All results saved to CSV files.")
print("\nAdditional files created:")
print("  - all_comparisons_to_period1.csv: Combined dataframe with all comparisons")
print("  - inboorling_semantic_change_aggregated.csv: Semantic changes for 'inboorling' (all variations)")
print("  - inboorling_neighbors_aggregated.txt: Nearest neighbors of 'inboorling' (aggregated) in all periods")
print("  - inboorling_temporal_change_aggregated.png: Plot of semantic change over time")
print("  - inboorling_changepoints_aggregated_*.png: Plots with detected changepoints")
print("  - changepoints-inboorling-aggregated-*.csv: Detected changepoints for specific parameters")
print("  - inboorling_changepoints_aggregated_all_parameters.csv: Changepoints for multiple parameter combinations")