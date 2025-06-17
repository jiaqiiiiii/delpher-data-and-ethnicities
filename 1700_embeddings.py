import os
import pandas as pd
import gensim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import re
import ruptures as rpt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from adjustText import adjust_text
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==================== UTILITY FUNCTIONS ====================

def safe_word_check(models, word, decade):
    """Unified function to safely check if word exists in model"""
    return (decade in models and 
            word in models.get(decade, {}).wv.key_to_index)

def get_word_neighbors(models, word, decade, topn=10, exclude_words=None):
    """Unified function to safely get word neighbors with optional exclusion"""
    if not safe_word_check(models, word, decade):
        return []
    
    if exclude_words is None:
        exclude_words = []
    
    try:
        # Get more neighbors than needed to account for exclusions
        extra_topn = min(topn * 3, len(models[decade].wv.key_to_index))
        neighbors = models[decade].wv.most_similar(word, topn=extra_topn)
        
        # Filter out excluded words
        filtered_neighbors = []
        for neighbor, similarity in neighbors:
            if neighbor.lower() not in [ex.lower() for ex in exclude_words]:
                filtered_neighbors.append(neighbor)
                if len(filtered_neighbors) >= topn:
                    break
        
        return filtered_neighbors
    except Exception as e:
        print(f"Error getting neighbors for '{word}' in {decade}: {e}")
        return []

def get_word_vector(models, word, decade):
    """Safely get word vector"""
    if safe_word_check(models, word, decade):
        return models[decade].wv[word]
    return None

def setup_plot(figsize=(12, 6), title="", xlabel="", ylabel=""):
    """Unified plot setup to avoid repetition"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig, ax

def calculate_distance(pos1, pos2):
    """Helper function for distance calculations"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def find_optimal_position(center_pos, occupied_positions, radii=[0.8, 1.2, 1.6, 2.0], angles=None):
    """
    Unified function for optimal label positioning used in both target and neighbor placement
    """
    if angles is None:
        angles = range(0, 360, 30)  # Default angles
    
    x, y = center_pos
    best_position = None
    best_score = 0
    
    for radius in radii:
        for angle in angles:
            rad = np.radians(angle)
            test_x = x + radius * np.cos(rad)
            test_y = y + radius * np.sin(rad)
            
            # Calculate minimum distance to all occupied positions
            min_dist = float('inf')
            for occ_x, occ_y, occ_radius in occupied_positions:
                dist = calculate_distance((test_x, test_y), (occ_x, occ_y))
                min_dist = min(min_dist, dist - occ_radius)
            
            if min_dist > best_score:
                best_score = min_dist
                best_position = (test_x, test_y)
    
    return best_position if best_position and best_score > 0.5 else (x + 1.5, y + 1.5)

# ==================== MAIN ANALYSIS FUNCTIONS ====================

# Define the same word groups as in your first script
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
    """
    Replace all spelling variations with their base words in the signature text
    """
    if not signature_text or pd.isna(signature_text):
        return ""
    
    # Split into words
    words = signature_text.split()
    normalized_words = []
    
    for word in words:
        # Clean the word (remove any remaining punctuation)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        # Replace with base word if it's a variation, otherwise keep original
        if clean_word in variation_to_base:
            normalized_words.append(variation_to_base[clean_word])
        else:
            normalized_words.append(word)  # Keep original case/form for other words
    
    return ' '.join(normalized_words)

# Step 1: Load and preprocess
print("Loading data...")
df = pd.read_csv('1700_unique_contexts_by_word.csv', sep=';')
print("Columns in df:", repr(df.columns.tolist()))
print(f"Original data shape: {df.shape}")

# Normalize signatures to replace variations with base words
print("Normalizing signatures...")
df['normalized_signature'] = df['signature'].apply(normalize_signature_for_embeddings)

# Check the normalization worked
print("\nSample of normalization:")
for i in range(min(5, len(df))):
    print(f"Original: {df.iloc[i]['signature']}")
    print(f"Normalized: {df.iloc[i]['normalized_signature']}")
    print(f"Base word: {df.iloc[i]['word']}")
    print("---")

# Tokenize the normalized signatures
df['tokenized_signature'] = df['normalized_signature'].str.split()

# Process years and decades
df['year'] = pd.to_datetime(df['year'], format="%Y", errors='coerce').dt.year
df = df.dropna(subset=['year', 'tokenized_signature'])
df['decade'] = (df['year'] // 10) * 10

print(f"Data shape after preprocessing: {df.shape}")
print(f"Decades found: {sorted(df['decade'].unique())}")

# Step 2: Train Word2Vec per decade
print("\nTraining Word2Vec models per decade...")
models = {}
for decade, grp in df.groupby('decade'):
    sentences = grp['tokenized_signature'].tolist()
    print(f"Decade {decade}: {len(sentences)} sentences")
    
    # Check if our target words appear in this decade's data
    all_words_in_decade = set()
    for sentence in sentences:
        all_words_in_decade.update(sentence)
    
    target_words_found = [word for word in ["moor", "inboorling"] if word in all_words_in_decade]
    print(f"  Target words found: {target_words_found}")
    
    model = gensim.models.Word2Vec(sentences, vector_size=300, window=3, min_count=5, sg=0, workers=4)
    models[decade] = model
    
    # Print vocabulary size and check for target words
    print(f"  Vocabulary size: {len(model.wv.key_to_index)}")
    for target_word in ["moor", "inboorling"]:
        if safe_word_check(models, target_word, decade):
            print(f"  '{target_word}' found in model with {len([s for s in sentences if target_word in s])} contexts")

# Step 3: Compute cosine similarities for target words across decades
print("\nComputing semantic similarities across decades...")
target_words = ["moor", "inboorling"]
records = []

for word in target_words:
    print(f"\nProcessing word: {word}")
    decades_with_word = [decade for decade in sorted(models.keys()) 
                        if safe_word_check(models, word, decade)]
    
    print(f"  Found in decades: {decades_with_word}")
    
    for dec in decades_with_word:
        vec = get_word_vector(models, word, dec)
        if vec is None:
            continue
            
        # Compare with vector of same word in every other decade
        sims = []
        compared_decades = []
        
        for dec2 in decades_with_word:
            if dec2 != dec:
                vec2 = get_word_vector(models, word, dec2)
                if vec2 is not None:
                    similarity = 1 - cosine(vec, vec2)
                    sims.append(similarity)
                    compared_decades.append(dec2)
        
        if sims:
            records.append({
                "word": word, 
                "decade": dec,
                "mean_sim": np.mean(sims),
                "std_sim": np.std(sims),
                "n_comparisons": len(sims),
                "compared_decades": compared_decades
            })
            print(f"  Decade {dec}: mean_sim={np.mean(sims):.3f}, std={np.std(sims):.3f}, n_comp={len(sims)}")

sim_df = pd.DataFrame(records)

# Print summary statistics
print(f"\nSimilarity analysis complete:")
print(f"Total records: {len(sim_df)}")
if not sim_df.empty:
    print("Records per word:")
    print(sim_df.groupby("word").size())
    
    # Also check word frequencies in the final dataset
    print(f"\nWord frequencies in training data:")
    all_tokens = []
    for sentences in [grp['tokenized_signature'].tolist() for _, grp in df.groupby('decade')]:
        for sentence in sentences:
            all_tokens.extend(sentence)
    
    word_counts = Counter(all_tokens)
    for target_word in target_words:
        count = word_counts.get(target_word, 0)
        print(f"  '{target_word}': {count} occurrences")

# Step 4: Create visualization
if sim_df.empty:
    print("No similarity data — check tokenization, min_count, and word frequency.")
else:
    print("\nCreating visualization...")
    fig, ax = setup_plot(
        figsize=(12, 6),
        title="Evolution of Semantic Stability by Word Over Time\n(After normalizing spelling variations)",
        xlabel="Decade",
        ylabel="Mean Cosine Similarity to other decades"
    )
    
    for word, grp in sim_df.groupby("word"):
        ax.errorbar(grp['decade'], grp['mean_sim'], yerr=grp['std_sim'], 
                   label=f"{word} (n={len(grp)})", capsize=3, marker='o', linewidth=2)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig('semantic_evolution_normalized.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed results
    sim_df.to_csv('semantic_similarity_results_normalized.csv', index=False)
    print("Results saved to 'semantic_similarity_results_normalized.csv'")

print("\nAnalysis complete!")

# ==================== TRAJECTORY VISUALIZATION ====================

def create_trajectory_visualization_data(models, target_word, topn_neighbors=8):
    """
    Extract and organize data for trajectory visualization (separated from plotting logic)
    """
    embeddings = []
    labels = []
    time_points = []
    target_word_positions = {}
    colors_map = {}
    
    # Create distinct color palette for different decades
    decades = sorted(models.keys())
    distinct_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'
    ]
    
    decade_colors = {decade: distinct_colors[i % len(distinct_colors)] 
                    for i, decade in enumerate(decades)}
    
    # Extract embeddings and nearest neighbors for each decade
    for decade in decades:
        try:
            # Get target word vector
            word_vector = get_word_vector(models, target_word, decade)
            if word_vector is None:
                continue
                
            embeddings.append(word_vector)
            labels.append(f"{target_word}_{decade}s")
            time_points.append(f"{decade}s")
            target_word_positions[f"{decade}s"] = len(embeddings) - 1
            colors_map[len(embeddings) - 1] = '#FF0000'  # Target word in bright red
            
            # Add nearest neighbors using unified function
            neighbors = get_word_neighbors(models, target_word, decade, topn_neighbors)
            for neighbor in neighbors:
                neighbor_vector = get_word_vector(models, neighbor, decade)
                if neighbor_vector is not None:
                    embeddings.append(neighbor_vector)
                    labels.append(neighbor)
                    time_points.append(f"{decade}s")
                    colors_map[len(embeddings) - 1] = decade_colors[decade]
                    
        except Exception as e:
            print(f"Error processing '{target_word}' in {decade}s: {e}")
            continue
    
    return {
        'embeddings': embeddings,
        'labels': labels,
        'time_points': time_points,
        'target_positions': target_word_positions,
        'colors_map': colors_map,
        'decade_colors': decade_colors
    }

def place_labels_optimally(target_coords, neighbor_coords):
    """
    Unified label placement function for both target and neighbor words
    """
    # Place target labels first
    placed_labels = []
    all_occupied_positions = []
    
    for x, y, label, idx in target_coords:
        position = find_optimal_position(
            (x, y), all_occupied_positions, 
            radii=[2.0, 2.5, 3.0, 3.5], angles=range(0, 360, 30)
        )
        placed_labels.append((position[0], position[1], label, idx))
        all_occupied_positions.append((position[0], position[1], 1.5))
    
    # Add target point positions to occupied positions
    for x, y, _, _ in target_coords:
        all_occupied_positions.append((x, y, 1.0))
    
    # Place neighbor labels
    neighbor_labels = []
    for x, y, label, color in neighbor_coords:
        position = find_optimal_position(
            (x, y), all_occupied_positions,
            radii=[0.8, 1.2, 1.6, 2.0], angles=range(0, 360, 45)
        )
        neighbor_labels.append((position[0], position[1], label, color, x, y))
        all_occupied_positions.append((position[0], position[1], 0.8))
    
    return placed_labels, neighbor_labels

def visualize_semantic_trajectory(models, target_word, topn_neighbors=8):
    """
    Visualize semantic change trajectory for a single target word across decades
    """
    # Get organized data
    data = create_trajectory_visualization_data(models, target_word, topn_neighbors)
    
    if len(data['embeddings']) == 0:
        print(f"No embeddings found for '{target_word}'")
        return
    
    # Convert embeddings to NumPy array and apply t-SNE
    embeddings_array = np.array(data['embeddings'])
    print(f"Total embeddings: {len(embeddings_array)}")
    
    perplexity_value = min(max(embeddings_array.shape[0] - 1, 2), 30)
    print(f"Using perplexity: {perplexity_value}")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, 
                n_iter=1000, learning_rate=200)
    reduced_embeddings = tsne.fit_transform(embeddings_array)
    
    # Create larger figure
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Separate target word coordinates and neighbor coordinates
    target_coords = []
    neighbor_coords = []
    
    for i, (x, y) in enumerate(reduced_embeddings):
        label = data['labels'][i]
        
        if label.startswith(target_word):  # Target word across decades
            target_coords.append((x, y, label, i))
            ax.scatter(x, y, color='#FF0000', s=250, edgecolors='black', 
                      linewidth=3, zorder=10, alpha=0.9)
        else:  # Neighbor words
            color = data['colors_map'].get(i, 'gray')
            ax.scatter(x, y, color=color, s=120, alpha=0.8, zorder=5,
                      edgecolors='white', linewidth=1.5)
            neighbor_coords.append((x, y, label, color))
    
    # Draw trajectory arrows between target word positions
    if len(target_coords) > 1:
        target_coords_sorted = sorted(target_coords, 
                                    key=lambda x: int(x[2].split('_')[1].replace('s', '')))
        
        for i in range(len(target_coords_sorted) - 1):
            start_pos = target_coords_sorted[i][:2]
            end_pos = target_coords_sorted[i + 1][:2]
            
            ax.annotate('', xy=end_pos, xytext=start_pos,
                       arrowprops=dict(arrowstyle='->', color='#FF0000', lw=5, 
                                     alpha=0.9, shrinkA=20, shrinkB=20))
    
    # Use unified label placement
    target_label_positions, neighbor_label_positions = place_labels_optimally(target_coords, neighbor_coords)
    
    # Draw target word labels with connecting lines
    target_texts = []
    for label_x, label_y, label, original_idx in target_label_positions:
        # Find original coordinates
        original_x, original_y = None, None
        for x, y, lbl, idx in target_coords:
            if idx == original_idx:
                original_x, original_y = x, y
                break
        
        # Create prominent label with background
        text = ax.text(label_x, label_y, label, fontsize=14, fontweight='bold', 
                      ha='center', va='center', color='#FF0000', zorder=15,
                      bbox=dict(boxstyle="round,pad=0.6", facecolor='white', 
                              alpha=0.95, edgecolor='#FF0000', linewidth=3))
        target_texts.append(text)
        
        # Draw connecting line
        if original_x is not None and original_y is not None:
            ax.plot([original_x, label_x], [original_y, label_y], 
                   color='#FF0000', linestyle='--', alpha=0.7, linewidth=2, zorder=12)
    
    # Draw neighbor labels
    neighbor_texts = []
    for text_x, text_y, label, color, orig_x, orig_y in neighbor_label_positions:
        text = ax.text(text_x, text_y, label, fontsize=10, ha='center', va='center', 
                      alpha=0.9, color='black', weight='normal', zorder=8,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                              alpha=0.9, edgecolor=color, linewidth=1.5))
        neighbor_texts.append(text)
        
        # Draw connecting line if needed
        distance = calculate_distance((text_x, text_y), (orig_x, orig_y))
        if distance > 0.5:
            ax.plot([orig_x, text_x], [orig_y, text_y], color=color, 
                   linestyle='-', alpha=0.5, linewidth=1, zorder=6)
    
    # Final adjustment using adjustText for any remaining overlaps
    all_texts = target_texts + neighbor_texts
    try:
        adjust_text(all_texts, ax=ax, expand_points=(1.5, 1.5), expand_text=(1.2, 1.2),
                   force_points=(0.3, 0.3), force_text=(0.8, 0.8), max_move=1.0,
                   precision=0.01, avoid_text=True, avoid_points=True, only_move={'text': 'xy'})
    except Exception as e:
        print(f"Final adjustText adjustment failed: {e}")
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0000', 
                  markersize=16, label=f'{target_word} (target word)', 
                  markeredgecolor='black', markeredgewidth=2)
    ]
    
    decades = sorted(models.keys())
    for decade in decades:
        if f"{decade}s" in data['time_points']:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=data['decade_colors'][decade], markersize=14, 
                          label=f'{decade}s neighbors',
                          markeredgecolor='white', markeredgewidth=1)
            )
    
    # Position legend and formatting
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', 
             framealpha=0.95, fontsize=12, fancybox=True, shadow=True)
    
    ax.set_title(f"Semantic Trajectory of '{target_word}' Over Time\n"
                f"Red dots show the target word, arrows show temporal progression", 
                fontsize=18, fontweight='bold', pad=30)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=16)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=16)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set appropriate margins
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_margin = (x_max - x_min) * 0.25
    y_margin = (y_max - y_min) * 0.25
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(f'semantic_trajectory_{target_word}_final.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return reduced_embeddings, data['labels'], data['time_points']

def visualize_all_target_words(models, target_words):
    """Create trajectory visualizations for all target words"""
    for word in target_words:
        print(f"\nCreating final trajectory visualization for '{word}'...")
        try:
            visualize_semantic_trajectory(models, word, topn_neighbors=6)
        except Exception as e:
            print(f"Error visualizing '{word}': {e}")

# Example usage:
visualize_all_target_words(models, ["moor", "inboorling"])

# ==================== CHANGEPOINT DETECTION ====================

def prepare_time_series_data(models, target_words, reference_decade=None):
    """Prepare time series data for changepoint detection"""
    time_series_data = {}
    
    for word in target_words:
        print(f"\nPreparing time series for '{word}'...")
        
        # Get all decades where the word appears using unified function
        decades_with_word = [decade for decade in sorted(models.keys()) 
                           if safe_word_check(models, word, decade)]
        
        if len(decades_with_word) < 3:
            print(f"  Warning: '{word}' appears in only {len(decades_with_word)} decades - skipping")
            continue
            
        print(f"  Found in decades: {decades_with_word}")
        
        # Choose reference decade
        ref_decade = (reference_decade if reference_decade and reference_decade in decades_with_word 
                     else decades_with_word[0])
        ref_vector = get_word_vector(models, word, ref_decade)
        
        # Compute similarities to reference decade
        similarities = []
        decades_list = []
        
        for decade in decades_with_word:
            if decade != ref_decade:
                current_vector = get_word_vector(models, word, decade)
                if current_vector is not None and ref_vector is not None:
                    similarity = 1 - cosine(ref_vector, current_vector)
                    similarities.append(similarity)
                    decades_list.append(decade)
        
        if len(similarities) >= 2:
            time_series_data[word] = {
                'decades': decades_list,
                'similarities': similarities,
                'reference_decade': ref_decade
            }
            print(f"  Time series length: {len(similarities)} points")
        else:
            print(f"  Not enough data points for changepoint detection")
    
    return time_series_data

def detect_changepoints_ruptures(time_series, method='pelt', penalty=1.0, min_size=2):
    """Detect changepoints using the ruptures library with various algorithms"""
    if len(time_series) < 4:
        return []
    
    signal = np.array(time_series).reshape(-1, 1)
    
    try:
        if method == 'pelt':
            algo = rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
            result = algo.predict(pen=penalty)
        elif method == 'binseg':
            algo = rpt.Binseg(model="l2", min_size=min_size).fit(signal)
            result = algo.predict(n_bkps=2)
        elif method == 'window':
            algo = rpt.Window(width=max(3, len(signal)//3), model="l2", min_size=min_size).fit(signal)
            result = algo.predict(pen=penalty)
        else:
            algo = rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
            result = algo.predict(pen=penalty)
        
        return [cp for cp in result if cp < len(time_series)]
        
    except Exception as e:
        print(f"    Error in changepoint detection: {e}")
        return []

def analyze_changepoints_comprehensive(time_series_data, methods=['pelt', 'binseg', 'window'], 
                                     penalties=[0.5, 1.0, 2.0]):
    """Comprehensive changepoint analysis using multiple methods and parameters"""
    changepoint_results = {}
    
    for word, data in time_series_data.items():
        print(f"\n=== Changepoint Analysis for '{word}' ===")
        decades = data['decades']
        similarities = data['similarities']
        ref_decade = data['reference_decade']
        
        print(f"Reference decade: {ref_decade}")
        print(f"Time series: {len(similarities)} points from {min(decades)} to {max(decades)}")
        
        word_results = {}
        
        for method in methods:
            for penalty in penalties:
                key = f"{method}_pen{penalty}"
                changepoints = detect_changepoints_ruptures(similarities, method=method, penalty=penalty)
                word_results[key] = changepoints
                
                if changepoints:
                    cp_decades = [decades[cp-1] if cp > 0 else decades[0] for cp in changepoints]
                    print(f"  {method} (penalty={penalty}): changepoints at positions {changepoints} -> decades {cp_decades}")
                else:
                    print(f"  {method} (penalty={penalty}): no changepoints detected")
        
        changepoint_results[word] = {
            'data': data,
            'results': word_results
        }
    
    return changepoint_results

def visualize_changepoints(changepoint_results, figsize=(15, 10)):
    """Visualize time series with detected changepoints for all words and methods"""
    n_words = len(changepoint_results)
    if n_words == 0:
        print("No data to visualize")
        return
    
    fig, axes = plt.subplots(n_words, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten() if n_words > 1 else [axes[0]]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for idx, (word, word_data) in enumerate(changepoint_results.items()):
        ax = axes[idx]
        
        decades = word_data['data']['decades']
        similarities = word_data['data']['similarities']
        ref_decade = word_data['data']['reference_decade']
        
        # Plot the time series
        ax.plot(decades, similarities, 'ko-', linewidth=2, markersize=6, 
                label=f'Similarity to {ref_decade}s', alpha=0.7)
        
        # Plot changepoints for different methods
        color_idx = 0
        methods_plotted = set()
        
        for method_key, changepoints in word_data['results'].items():
            if changepoints:
                method_name = method_key.split('_')[0]
                if method_name not in methods_plotted:
                    color = colors[color_idx % len(colors)]
                    
                    for cp in changepoints:
                        if cp < len(decades):
                            cp_decade = decades[cp]
                            ax.axvline(x=cp_decade, color=color, linestyle='--', 
                                     alpha=0.8, linewidth=2, 
                                     label=f'{method_name} changepoint' if method_name not in methods_plotted else "")
                    
                    methods_plotted.add(method_name)
                    color_idx += 1
        
        ax.set_title(f"Semantic Change Timeline: '{word}'", fontsize=12, fontweight='bold')
        ax.set_xlabel("Decade")
        ax.set_ylabel(f"Cosine Similarity\n(to {ref_decade}s)")
        ax.legend()
        
        # Add annotation with statistics
        ax.text(0.02, 0.98, f"Mean: {np.mean(similarities):.3f}\nStd: {np.std(similarities):.3f}", 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('changepoint_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def statistical_changepoint_validation(changepoint_results):
    """Validate changepoints using statistical tests"""
    validation_results = {}
    
    for word, word_data in changepoint_results.items():
        print(f"\n=== Statistical Validation for '{word}' ===")
        
        decades = word_data['data']['decades']
        similarities = word_data['data']['similarities']
        
        # Find most consistent changepoints across methods
        all_changepoints = []
        for method_key, changepoints in word_data['results'].items():
            all_changepoints.extend(changepoints)
        
        if not all_changepoints:
            print("  No changepoints to validate")
            continue
        
        # Count frequency of each changepoint position
        cp_counts = Counter(all_changepoints)
        
        # Consider changepoints that appear in multiple methods
        consensus_changepoints = [cp for cp, count in cp_counts.items() if count >= 2]
        
        validation_results[word] = {
            'all_changepoints': all_changepoints,
            'consensus_changepoints': consensus_changepoints,
            'statistical_tests': {}
        }
        
        print(f"  All detected changepoints: {sorted(set(all_changepoints))}")
        print(f"  Consensus changepoints (≥2 methods): {consensus_changepoints}")
        
        # Perform statistical tests for consensus changepoints
        for cp in consensus_changepoints:
            if 0 < cp < len(similarities):
                before = similarities[:cp]
                after = similarities[cp:]
                
                if len(before) >= 2 and len(after) >= 2:
                    # t-test for mean difference
                    t_stat, t_pval = stats.ttest_ind(before, after)
                    
                    # Mann-Whitney U test (non-parametric)
                    u_stat, u_pval = stats.mannwhitneyu(before, after, alternative='two-sided')
                    
                    cp_decade = decades[cp] if cp < len(decades) else "end"
                    
                    validation_results[word]['statistical_tests'][cp] = {
                        'decade': cp_decade,
                        'before_mean': np.mean(before),
                        'after_mean': np.mean(after),
                        'before_std': np.std(before),
                        'after_std': np.std(after),
                        't_test': {'statistic': t_stat, 'p_value': t_pval},
                        'mann_whitney': {'statistic': u_stat, 'p_value': u_pval}
                    }
                    
                    print(f"  Changepoint at decade {cp_decade}:")
                    print(f"    Before: mean={np.mean(before):.3f}, std={np.std(before):.3f}")
                    print(f"    After:  mean={np.mean(after):.3f}, std={np.std(after):.3f}")
                    print(f"    t-test p-value: {t_pval:.4f}")
                    print(f"    Mann-Whitney p-value: {u_pval:.4f}")
    
    return validation_results

def create_changepoint_summary_table(validation_results):
    """Create a summary table of all changepoints and their statistical significance"""
    summary_data = []
    
    for word, results in validation_results.items():
        for cp, stats_data in results.get('statistical_tests', {}).items():
            summary_data.append({
                'Word': word,
                'Changepoint_Position': cp,
                'Decade': stats_data['decade'],
                'Before_Mean': stats_data['before_mean'],
                'After_Mean': stats_data['after_mean'],
                'Mean_Difference': stats_data['after_mean'] - stats_data['before_mean'],
                'T_Test_P_Value': stats_data['t_test']['p_value'],
                'Mann_Whitney_P_Value': stats_data['mann_whitney']['p_value'],
                'Significant_005': stats_data['t_test']['p_value'] < 0.05
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.round(4)
        
        print("\n=== CHANGEPOINT SUMMARY TABLE ===")
        print(summary_df.to_string(index=False))
        
        # Save to CSV
        summary_df.to_csv('changepoint_summary.csv', index=False)
        print("\nSummary saved to 'changepoint_summary.csv'")
        
        return summary_df
    else:
        print("No changepoints found for summary table")
        return pd.DataFrame()

# ===== MAIN EXECUTION =====
print("\n" + "="*60)
print("STARTING CHANGEPOINT DETECTION ANALYSIS")
print("="*60)

# Step 1: Prepare time series data
print("\nStep 1: Preparing time series data...")
time_series_data = prepare_time_series_data(models, target_words)

if not time_series_data:
    print("No suitable time series data found. Check that your target words appear in multiple decades.")
else:
    # Step 2: Comprehensive changepoint analysis
    print("\nStep 2: Running comprehensive changepoint analysis...")
    changepoint_results = analyze_changepoints_comprehensive(
        time_series_data, 
        methods=['pelt', 'binseg'], 
        penalties=[0.5, 1.0, 2.0]
    )
    
    # Step 3: Visualize results
    print("\nStep 3: Creating visualizations...")
    visualize_changepoints(changepoint_results)
    
    # Step 4: Statistical validation
    print("\nStep 4: Statistical validation of changepoints...")
    validation_results = statistical_changepoint_validation(changepoint_results)
    
    # Step 5: Create summary table
    print("\nStep 5: Creating summary table...")
    summary_df = create_changepoint_summary_table(validation_results)
    
    print("\n" + "="*60)
    print("CHANGEPOINT DETECTION ANALYSIS COMPLETE")
    print("="*60)
    
    # Additional insights
    if not summary_df.empty:
        significant_changes = summary_df[summary_df['Significant_005'] == True]
        if len(significant_changes) > 0:
            print(f"\nFound {len(significant_changes)} statistically significant changepoints (p < 0.05):")
            for _, row in significant_changes.iterrows():
                print(f"  - '{row['Word']}' around {row['Decade']}: "
                      f"similarity changed from {row['Before_Mean']:.3f} to {row['After_Mean']:.3f}")
        else:
            print("\nNo statistically significant changepoints detected.")
    
    print("\nFiles generated:")
    print("  - changepoint_detection_results.png")
    print("  - changepoint_summary.csv")
    print("  - semantic_similarity_results_normalized.csv (from previous analysis)")

# ==================== NEIGHBOR ANALYSIS ====================

def analyze_cooccurrence_statistics(df, target_words):
    """Analyze how often target words co-occur in the same sentences"""
    print("\n" + "="*60)
    print("CO-OCCURRENCE ANALYSIS")
    print("="*60)
    
    cooccurrence_stats = {}
    total_sentences = len(df)
    
    for i, word1 in enumerate(target_words):
        for j, word2 in enumerate(target_words):
            if i < j:  # Avoid duplicate pairs
                both_present = 0
                word1_present = 0
                word2_present = 0
                
                for sentences in df['tokenized_signature']:
                    has_word1 = word1 in sentences
                    has_word2 = word2 in sentences
                    
                    if has_word1:
                        word1_present += 1
                    if has_word2:
                        word2_present += 1
                    if has_word1 and has_word2:
                        both_present += 1
                
                cooccurrence_rate = both_present / total_sentences if total_sentences > 0 else 0
                word1_rate = word1_present / total_sentences if total_sentences > 0 else 0
                word2_rate = word2_present / total_sentences if total_sentences > 0 else 0
                
                # Calculate conditional probability
                conditional_prob = both_present / word1_present if word1_present > 0 else 0
                
                cooccurrence_stats[f"{word1}-{word2}"] = {
                    'both_present': both_present,
                    'cooccurrence_rate': cooccurrence_rate,
                    'word1_rate': word1_rate,
                    'word2_rate': word2_rate,
                    'conditional_prob': conditional_prob
                }
                
                print(f"\n📊 {word1.upper()} & {word2.upper()}:")
                print(f"   Co-occurrence: {both_present}/{total_sentences} sentences ({cooccurrence_rate:.1%})")
                print(f"   {word1} appears in: {word1_present}/{total_sentences} sentences ({word1_rate:.1%})")
                print(f"   {word2} appears in: {word2_present}/{total_sentences} sentences ({word2_rate:.1%})")
                print(f"   When {word1} appears, {word2} also appears: {conditional_prob:.1%}")
    
    return cooccurrence_stats

def analyze_semantic_shift_improved(before_neighbors, after_neighbors, before_decades, after_decades, 
                                  target_word, all_target_words):
    """
    Improved semantic shift analysis that excludes mutual target word influences
    """
    # Create exclusion list (other target words)
    exclude_words = [w for w in all_target_words if w.lower() != target_word.lower()]
    
    # Flatten all neighbors from before and after periods, excluding target words
    all_before = []
    all_after = []
    
    for decade, neighbors in before_neighbors.items():
        filtered_neighbors = [n for n in neighbors if n.lower() not in [ex.lower() for ex in exclude_words]]
        all_before.extend(filtered_neighbors)
    
    for decade, neighbors in after_neighbors.items():
        filtered_neighbors = [n for n in neighbors if n.lower() not in [ex.lower() for ex in exclude_words]]
        all_after.extend(filtered_neighbors)
    
    # Count frequency of neighbors
    before_counts = Counter(all_before)
    after_counts = Counter(all_after)
    
    # Find neighbors that appear frequently in each period
    before_frequent = set([word for word, count in before_counts.most_common(15)])
    after_frequent = set([word for word, count in after_counts.most_common(15)])
    
    # Analyze the shift
    moving_away_from = before_frequent - after_frequent  # In before but not after
    moving_towards = after_frequent - before_frequent    # In after but not before
    stable_context = before_frequent & after_frequent    # In both periods
    
    # Calculate shift intensity
    total_unique_neighbors = len(before_frequent | after_frequent)
    shift_intensity = len(moving_away_from | moving_towards) / total_unique_neighbors if total_unique_neighbors > 0 else 0
    
    return {
        'moving_away_from': list(moving_away_from),
        'moving_towards': list(moving_towards),
        'stable_context': list(stable_context),
        'before_frequent': dict(before_counts.most_common(10)),
        'after_frequent': dict(after_counts.most_common(10)),
        'shift_intensity': shift_intensity,
        'excluded_words': exclude_words
    }

def print_shift_analysis_improved(word, changepoint_decade, analysis):
    """Print an improved formatted analysis of the semantic shift"""
    print(f"\n  📊 IMPROVED SEMANTIC SHIFT ANALYSIS for '{word}' around {changepoint_decade}s:")
    print(f"  🚫 Excluded from analysis: {', '.join(analysis['excluded_words'])}")
    print(f"  📈 Shift intensity: {analysis['shift_intensity']:.1%}")
    
    moving_away = analysis['moving_away_from']
    moving_towards = analysis['moving_towards']
    stable = analysis['stable_context']
    
    if moving_away:
        print(f"  🔄 Moving AWAY from ({len(moving_away)} contexts): {', '.join(moving_away[:10])}")
    else:
        print(f"  🔄 Moving AWAY from: [no clear pattern]")
    
    if moving_towards:
        print(f"  ➡️  Moving TOWARDS ({len(moving_towards)} contexts): {', '.join(moving_towards[:10])}")
    else:
        print(f"  ➡️  Moving TOWARDS: [no clear pattern]")
    
    if stable:
        print(f"  🔒 Stable context ({len(stable)} contexts): {', '.join(stable[:8])}")
    else:
        print(f"  🔒 Stable context: [limited overlap]")

def analyze_direct_word_similarity(models, target_words):
    """Analyze direct cosine similarity between target words across decades"""
    print("\n" + "="*60)
    print("DIRECT WORD SIMILARITY ANALYSIS")
    print("="*60)
    
    similarity_data = []
    
    for i, word1 in enumerate(target_words):
        for j, word2 in enumerate(target_words):
            if i < j:  # Avoid duplicate pairs
                print(f"\n📊 Direct similarity: {word1.upper()} ↔ {word2.upper()}")
                
                for decade in sorted(models.keys()):
                    vec1 = get_word_vector(models, word1, decade)
                    vec2 = get_word_vector(models, word2, decade)
                    
                    if vec1 is not None and vec2 is not None:
                        similarity = 1 - cosine(vec1, vec2)
                        similarity_data.append({
                            'decade': decade,
                            'word1': word1,
                            'word2': word2,
                            'similarity': similarity
                        })
                        print(f"   {decade}s: {similarity:.3f}")
    
    return pd.DataFrame(similarity_data)

def analyze_all_changepoints_neighbors_improved(models, changepoint_results, validation_results, 
                                              target_words, topn=10, context_window=1, min_methods=1):
    """
    IMPROVED neighbor analysis that addresses mutual influence issues
    """
    
    neighbor_analysis = {}
    
    for word in changepoint_results.keys():
        print(f"\n{'='*60}")
        print(f"IMPROVED NEAREST NEIGHBORS ANALYSIS FOR '{word.upper()}'")
        print(f"{'='*60}")
        
        # Get the decades and changepoints for this word
        decades = changepoint_results[word]['data']['decades']
        similarities = changepoint_results[word]['data']['similarities']
        
        # First try consensus changepoints
        consensus_changepoints = validation_results.get(word, {}).get('consensus_changepoints', [])
        
        # If no consensus, use all changepoints that appear in at least min_methods
        if not consensus_changepoints:
            print(f"No consensus changepoints found for '{word}'. Using all detected changepoints...")
            
            # Get all changepoints from all methods
            all_changepoints_data = validation_results.get(word, {}).get('all_changepoints', [])
            
            if not all_changepoints_data:
                print(f"No changepoints found at all for '{word}'. SKIPPING.")
                continue
            
            # Count changepoint occurrences
            changepoint_counts = Counter(all_changepoints_data)
            
            # Use changepoints detected by at least min_methods
            working_changepoints = [cp for cp, count in changepoint_counts.items() 
                                  if count >= min_methods]
            
            print(f"  Changepoint counts: {changepoint_counts}")
            print(f"  Using changepoints detected by ≥{min_methods} methods: {working_changepoints}")
        else:
            working_changepoints = consensus_changepoints
            print(f"Using consensus changepoints: {working_changepoints}")
        
        if not working_changepoints:
            print(f"No valid changepoints found for '{word}'. SKIPPING.")
            continue
        
        word_analysis = {}
        
        for cp_idx in working_changepoints:
            if cp_idx >= len(decades):
                print(f"  Changepoint index {cp_idx} >= decades length {len(decades)}. SKIPPING.")
                continue
                
            cp_decade = decades[cp_idx]
            print(f"\n--- Changepoint around {cp_decade}s (index {cp_idx}) ---")
            
            # Define the decades to analyze around the changepoint
            all_decades = sorted(models.keys())
            cp_decade_idx = all_decades.index(cp_decade) if cp_decade in all_decades else -1
            
            if cp_decade_idx == -1:
                print(f"  Decade {cp_decade} not found in models. SKIPPING.")
                continue
            
            # Get before and after decades within the context window
            before_decades = []
            after_decades = []
            
            # Before changepoint
            for i in range(max(0, cp_decade_idx - context_window), cp_decade_idx):
                if safe_word_check(models, word, all_decades[i]):
                    before_decades.append(all_decades[i])
            
            # After changepoint  
            for i in range(cp_decade_idx, min(len(all_decades), cp_decade_idx + context_window + 1)):
                if safe_word_check(models, word, all_decades[i]):
                    after_decades.append(all_decades[i])
            
            print(f"Analyzing decades before: {before_decades}")
            print(f"Analyzing decades after: {after_decades}")
            
            if not before_decades and not after_decades:
                print(f"  No valid decades found around changepoint. SKIPPING.")
                continue
            
            # Get neighbors for before and after periods using IMPROVED function with exclusions
            exclude_words = [w for w in target_words if w.lower() != word.lower()]
            before_neighbors = {}
            after_neighbors = {}
            
            print(f"🚫 Excluding from neighbor analysis: {exclude_words}")
            
            # Collect neighbors from before decades
            for decade in before_decades:
                neighbors = get_word_neighbors(models, word, decade, topn, exclude_words)
                if neighbors:
                    before_neighbors[decade] = neighbors
                    print(f"  {decade}s neighbors (filtered): {', '.join(neighbors[:8])}")
            
            # Collect neighbors from after decades
            for decade in after_decades:
                neighbors = get_word_neighbors(models, word, decade, topn, exclude_words)
                if neighbors:
                    after_neighbors[decade] = neighbors
                    print(f"  {decade}s neighbors (filtered): {', '.join(neighbors[:8])}")
            
            # Only proceed if we have neighbors from at least one period
            if not before_neighbors and not after_neighbors:
                print(f"  No neighbors found for any period. SKIPPING.")
                continue
            
            # Analyze the shift using IMPROVED function
            cp_analysis = analyze_semantic_shift_improved(before_neighbors, after_neighbors, 
                                                        before_decades, after_decades, 
                                                        word, target_words)
            
            word_analysis[cp_decade] = {
                'changepoint_index': cp_idx,
                'before_decades': before_decades,
                'after_decades': after_decades,
                'before_neighbors': before_neighbors,
                'after_neighbors': after_neighbors,
                'shift_analysis': cp_analysis
            }
            
            # Print the IMPROVED shift analysis
            print_shift_analysis_improved(word, cp_decade, cp_analysis)
        
        if word_analysis:
            neighbor_analysis[word] = word_analysis
            print(f"\n✅ Completed improved analysis for '{word}': {len(word_analysis)} changepoints analyzed")
        else:
            print(f"\n❌ No valid changepoints analyzed for '{word}'")
    
    return neighbor_analysis

def save_neighbor_analysis_to_file(neighbor_analysis, filename='neighbor_analysis_results.txt'):
    """Save the neighbor analysis results to a text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("SEMANTIC NEIGHBOR ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")
        
        for word, word_analysis in neighbor_analysis.items():
            f.write(f"WORD: {word.upper()}\n")
            f.write("-" * 40 + "\n")
            
            for changepoint_decade, cp_data in word_analysis.items():
                f.write(f"\nChangepoint around {changepoint_decade}s:\n")
                f.write(f"  Index: {cp_data['changepoint_index']}\n")
                f.write(f"  Before decades: {cp_data['before_decades']}\n")
                f.write(f"  After decades: {cp_data['after_decades']}\n\n")
                
                # Neighbors by decade
                f.write("  Neighbors by decade:\n")
                for decade, neighbors in cp_data['before_neighbors'].items():
                    f.write(f"    {decade}s: {', '.join(neighbors[:8])}\n")
                for decade, neighbors in cp_data['after_neighbors'].items():
                    f.write(f"    {decade}s: {', '.join(neighbors[:8])}\n")
                
                # Semantic shift analysis
                shift = cp_data['shift_analysis']
                f.write(f"\n  SEMANTIC SHIFT ANALYSIS:\n")
                
                if shift['moving_away_from']:
                    f.write(f"    Moving AWAY from: {', '.join(shift['moving_away_from'][:10])}\n")
                else:
                    f.write(f"    Moving AWAY from: [no clear pattern]\n")
                
                if shift['moving_towards']:
                    f.write(f"    Moving TOWARDS: {', '.join(shift['moving_towards'][:10])}\n")
                else:
                    f.write(f"    Moving TOWARDS: [no clear pattern]\n")
                
                if shift['stable_context']:
                    f.write(f"    Stable context: {', '.join(shift['stable_context'][:8])}\n")
                else:
                    f.write(f"    Stable context: [limited overlap]\n")
                
                f.write("\n" + "-" * 40 + "\n")
            
            f.write("\n" + "="*60 + "\n\n")
    
    print(f"Neighbor analysis results saved to '{filename}'")

def create_detailed_neighbor_summary(neighbor_analysis):
    """Create a detailed summary of all neighbor analysis results"""
    print("\n" + "="*80)
    print("DETAILED NEIGHBOR ANALYSIS SUMMARY")
    print("="*80)
    
    total_changepoints = 0
    
    for word, word_analysis in neighbor_analysis.items():
        print(f"\n📊 SUMMARY FOR '{word.upper()}':")
        print(f"   Total changepoints analyzed: {len(word_analysis)}")
        total_changepoints += len(word_analysis)
        
        for changepoint_decade, cp_data in word_analysis.items():
            print(f"\n   🔄 Changepoint around {changepoint_decade}s:")
            shift = cp_data['shift_analysis']
            
            # Count the semantic changes
            away_count = len(shift['moving_away_from'])
            towards_count = len(shift['moving_towards'])
            stable_count = len(shift['stable_context'])
            
            print(f"     • Moving away from {away_count} contexts")
            print(f"     • Moving towards {towards_count} contexts") 
            print(f"     • Stable in {stable_count} contexts")
            
            # Show top contexts
            if shift['moving_away_from']:
                print(f"     • Key departures: {', '.join(shift['moving_away_from'][:5])}")
            if shift['moving_towards']:
                print(f"     • Key arrivals: {', '.join(shift['moving_towards'][:5])}")
            if shift['stable_context']:
                print(f"     • Stable core: {', '.join(shift['stable_context'][:5])}")
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   Total words analyzed: {len(neighbor_analysis)}")
    print(f"   Total changepoints found: {total_changepoints}")
    print(f"   Average changepoints per word: {total_changepoints/len(neighbor_analysis):.1f}")

# ===== MAIN EXECUTION FOR NEIGHBOR ANALYSIS =====
print("\n" + "="*60)
print("RUNNING IMPROVED NEAREST NEIGHBORS ANALYSIS")
print("="*60)

# Step 0: Analyze co-occurrence patterns
print("\nStep 0: Analyzing co-occurrence patterns...")
cooccurrence_stats = analyze_cooccurrence_statistics(df, target_words)

# Step 1: Analyze direct word similarity
print("\nStep 1: Analyzing direct word similarities...")
direct_similarity_df = analyze_direct_word_similarity(models, target_words)
if not direct_similarity_df.empty:
    direct_similarity_df.to_csv('direct_word_similarities.csv', index=False)
    print("Direct similarity data saved to 'direct_word_similarities.csv'")

# Run the IMPROVED neighbor analysis
if 'changepoint_results' in locals() and 'validation_results' in locals():
    print("\nStep 2: Analyzing nearest neighbors around all detected changepoints (IMPROVED)...")
    neighbor_analysis = analyze_all_changepoints_neighbors_improved(
        models, changepoint_results, validation_results, target_words, 
        topn=15, context_window=1, min_methods=1
    )
    
    if neighbor_analysis and len(neighbor_analysis) > 0:
        print(f"\n✅ Successfully analyzed {len(neighbor_analysis)} words with improved method")
        
        # Create detailed summary
        create_detailed_neighbor_summary(neighbor_analysis)
        
        # Save results to file
        save_neighbor_analysis_to_file(neighbor_analysis, 'neighbor_analysis_results_improved.txt')
        
        print("\n📁 FILES GENERATED:")
        print("   - neighbor_analysis_results_improved.txt (detailed text output)")
        print("   - direct_word_similarities.csv (word-to-word similarity data)")
        print("   - changepoint_detection_results.png (visualization)")
        print("   - changepoint_summary.csv (statistical summary)")
        print("   - semantic_similarity_results_normalized.csv (similarity data)")
        
        # Compare with original analysis if available
        print("\n🔍 COMPARISON WITH ORIGINAL:")
        print("   The improved analysis should show more distinct patterns")
        print("   by excluding mutual target word influences from neighbor lists.")
        
    else:
        print("\n❌ No valid analysis results generated.")
        print("This might mean:")
        print("1. No changepoints were detected by any method")
        print("2. The target words don't exist in the models")
        print("3. There's insufficient context around changepoints")
        print("4. After filtering, insufficient neighbors remain")
else:
    print("ERROR: changepoint_results or validation_results not found!")
    print("Make sure you've run the changepoint detection analysis first.")

print("\n" + "="*60)
print("ALL IMPROVED ANALYSES COMPLETE!")
print("="*60)