import os
import glob
import re
import pandas as pd
from bs4 import BeautifulSoup
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm  # For progress tracking
import unicodedata

# Configuration
target_words = ["moor", "moren", "moorin", "inboorling", "inboorlingen"]
base_dir = "/data/groups/trifecta/jiaqiz/downloaded_zip_delpher/kranten_pd_17xx"
window_size = 25  # 25 tokens before and after the target word
similarity_threshold = 0.7

def normalize_old_dutch_text(text):
    """Normalize 18th century Dutch text with OCR errors and spelling variations"""
    if not text:
        return ""
    
    # Convert to lowercase for processing
    text = text.lower()
    
    # Normalize unicode characters (handles accented characters)
    text = unicodedata.normalize('NFKD', text)
    
    # Common 17th century Dutch spelling normalizations
    replacements = {
        # Double letters that were common in old Dutch
        'ck': 'k',
        'gh': 'g',
        'tgh': 'tg',
        'ngh': 'ng',
        
        # Common letter substitutions in old Dutch/OCR errors
        'y': 'ij',  # y was often used instead of ij
        'ae': 'aa',  # ae -> aa
        
        # OCR common errors
        'rn': 'm',   # OCR often confuses rn with m
        'cl': 'd',   # OCR confusion
        'ii': 'ij',  # double i to ij
        
        # Remove common OCR artifacts
        '~': '',
        '^': '',
        '|': '',
        
        # Old Dutch specific
        'cq': 'kw',
        'x': 'ks',
    }
    
    # Apply replacements carefully
    for old, new in replacements.items():
        text = re.sub(r'\b' + old + r'\b', new, text)
    
    # Clean up whitespace and punctuation
    # Remove excessive punctuation but keep basic structure
    text = re.sub(r'[^\w\s.,;:!?-]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove standalone single characters that are likely OCR errors
    text = re.sub(r'\b[a-z]\b', ' ', text)
    
    return text.strip()

def clean_context_for_display(context):
    """Clean context for better readability while preserving important elements"""
    if not context:
        return ""
    
    # Remove excessive whitespace
    context = re.sub(r'\s+', ' ', context.strip())
    
    # Remove obvious OCR artifacts at word boundaries
    context = re.sub(r'\b[^\w\s]{2,}\b', ' ', context)
    
    # Clean up but preserve the general structure
    context = re.sub(r'[^\w\s.,;:!?()-]', ' ', context)
    context = re.sub(r'\s+', ' ', context)
    
    return context.strip()

def get_signature(text):
    """Generate a normalized signature for deduplication"""
    # More aggressive normalization for deduplication
    text = normalize_old_dutch_text(text)
    
    # Remove all punctuation for signature
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove common Dutch stop words for better similarity detection
    dutch_stop_words = ['de', 'het', 'een', 'en', 'van', 'in', 'is', 'dat', 'op', 'te', 'met', 'voor', 'door', 'om', 'tot', 'over', 'uit', 'aan', 'bij', 'onder', 'tegen', 'naar', 'binnen', 'buiten']
    words = text.split()
    words = [w for w in words if w not in dutch_stop_words and len(w) > 2]
    
    return ' '.join(words)

def get_hash(text):
    """Generate MD5 hash for exact duplicate detection"""
    normalized_text = get_signature(text)
    return hashlib.md5(normalized_text.encode('utf-8')).hexdigest()

def find_target_words_in_text(text, year, file_path):
    """Find target words in text with proper context extraction"""
    results = []
    
    # Store original text for reference
    original_text = text
    
    # Normalize the ENTIRE text first
    normalized_text = normalize_old_dutch_text(text)
    
    # Split normalized text into words - this is our working array
    normalized_words = normalized_text.split()
    
    # Create a mapping to find original positions
    # We'll work entirely with normalized text and then reconstruct context
    
    # Process each word in the normalized text
    for i, token in enumerate(normalized_words):
        # Clean token for matching (remove any remaining punctuation)
        clean_token = re.sub(r'[^\w]', '', token.lower())
        
        for target_word in target_words:
            # Check for exact match or close variations
            if (clean_token == target_word.lower() or 
                # Handle common variations
                (target_word == 'moor' and clean_token in ['moor', 'mooren', 'moore']) or
                (target_word == 'moren' and clean_token in ['moren', 'mooren', 'moore']) or
                (target_word == 'inboorling' and clean_token in ['inboorling', 'inboorlingh', 'inbooring']) or
                (target_word == 'inboorlingen' and clean_token in ['inboorlingen', 'inboorlinghen', 'inboorings'])):
                
                # Calculate start and end positions with exact window_size
                start = max(i - window_size, 0)
                end = min(i + window_size + 1, len(normalized_words))
                
                # Extract context from normalized words (this ensures exact positioning)
                context_words = normalized_words[start:end]
                context = " ".join(context_words)
                
                # Clean the context for display
                context = clean_context_for_display(context)
                
                # Find the target word position in the context for verification
                target_position = i - start  # Position of target word in the context
                
                # Store the result
                results.append({
                    "word": target_word,
                    "context": context,
                    "year": year,
                    "file_path": file_path,
                    "original_token": normalized_words[i],  # The normalized form that matched
                    "target_position": target_position,  # Position in context (for debugging)
                    "context_length": len(context_words),  # Total context length (for debugging)
                    "signature": get_signature(context),
                    "hash": get_hash(context)
                })
    
    return results

def remove_similar_contexts(contexts_df, threshold=0.7):
    """Remove similar contexts using cosine similarity with improved Dutch stop words"""
    print(f"Removing similar contexts for threshold {threshold}...")
    
    # If we have very few contexts, skip similarity check
    if len(contexts_df) <= 1:
        return contexts_df
    
    # Extract contexts for similarity comparison
    contexts = contexts_df['context'].tolist()
    
    # Enhanced Dutch stop words for 17th century texts
    dutch_stop_words = [
        'de', 'het', 'een', 'en', 'van', 'in', 'is', 'dat', 'op', 'te', 'met', 'voor', 
        'door', 'om', 'tot', 'over', 'uit', 'aan', 'bij', 'onder', 'tegen', 'naar', 
        'binnen', 'buiten', 'als', 'dan', 'maar', 'zo', 'zijn', 'haar', 'hem', 'hun',
        'die', 'dit', 'deze', 'der', 'des', 'den', 'ter', 'ten'  # Old Dutch articles
    ]
    
    # Create TF-IDF vectors with enhanced parameters for better similarity detection
    vectorizer = TfidfVectorizer(
        stop_words=dutch_stop_words,
        ngram_range=(1, 2),  # Include bigrams for better context
        min_df=1,  # Keep all terms
        max_features=5000,  # Limit features for performance
        analyzer='word'
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(contexts)
    except ValueError:
        # If vectorization fails, just return the original dataframe
        print("Vectorization failed, skipping similarity check")
        return contexts_df
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Create a mask of contexts to keep
    to_keep = np.ones(len(contexts), dtype=bool)
    
    # Iterate through each context
    for i in range(len(contexts)):
        if not to_keep[i]:
            continue  # Skip if already marked for removal
        
        # Find similar contexts
        for j in range(i + 1, len(contexts)):
            if to_keep[j] and similarity_matrix[i, j] > threshold:
                # Mark the similar context for removal
                # Keep the one with more meaningful content (longer after removing stop words)
                clean_i = get_signature(contexts[i])
                clean_j = get_signature(contexts[j])
                
                if len(clean_i) >= len(clean_j):
                    to_keep[j] = False
                else:
                    to_keep[i] = False
                    break  # Move to the next i
    
    # Filter the dataframe
    filtered_df = contexts_df[to_keep].copy()
    
    print(f"Removed {len(contexts_df) - len(filtered_df)} similar contexts")
    return filtered_df

def main():
    all_results = []
    
    # Process multiple century directories
    for century in ["17xx"]:
        century_dir = base_dir.replace("17xx", century)
        
        # Skip if directory doesn't exist
        if not os.path.exists(century_dir):
            print(f"Directory {century_dir} does not exist, skipping...")
            continue
            
        print(f"Processing {century_dir}...")
        
        # Collect all XML files in this century directory
        xml_files = []
        for root, dirs, files in os.walk(century_dir):
            for file in files:
                if file.endswith(".xml"):
                    xml_files.append(os.path.join(root, file))
        
        print(f"Found {len(xml_files)} XML files in {century_dir}")
        
        # Process each file with progress bar
        for file_path in tqdm(xml_files, desc=f"Processing {century} files"):
            # Extract year from directory structure
            year_match = re.search(r"/(\d{4})/", file_path)
            year = year_match.group(1) if year_match else "unknown"
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f, "xml")
                    text = soup.get_text(separator=' ', strip=True)
                    
                    if not text.strip():
                        continue
                    
                    # Find target words in text
                    file_results = find_target_words_in_text(text, year, file_path)
                    all_results.extend(file_results)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Create DataFrame
    print("Creating DataFrame...")
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("No results found!")
        return
    
    # Remove exact duplicates first
    initial_count = len(df)
    df = df.drop_duplicates(subset=['hash'])
    print(f"Removed {initial_count - len(df)} exact duplicates")
    
    # Group by word and remove similar contexts within each group
    final_df = pd.DataFrame()
    for word, group in df.groupby('word'):
        print(f"Processing {len(group)} contexts for '{word}'...")
        filtered_group = remove_similar_contexts(group, similarity_threshold)
        final_df = pd.concat([final_df, filtered_group], ignore_index=True)
    
    # Reset index and drop unnecessary columns for final output
    final_df = final_df.reset_index(drop=True)
    
    # Keep debugging columns for now to verify the fix
    print(f"Total unique contexts found: {len(final_df)}")
    
    # Save with semicolon separator to match your original format
    final_df.to_csv("unique_contexts_by_word.csv", index=False, encoding="utf-8", sep=';')
    print("Results saved to unique_contexts_by_word.csv")
    
    # Save a summary by word and year
    summary = final_df.groupby(['word', 'year']).size().reset_index(name='count')
    summary.to_csv("contexts_summary.csv", index=False)
    print("Summary saved to contexts_summary.csv")
    
    # Print summary statistics
    print("\nSummary by word:")
    word_summary = final_df['word'].value_counts()
    for word, count in word_summary.items():
        print(f"  {word}: {count} contexts")
    
    # Debug: Check if target words are properly centered
    print("\nDebugging target word positions:")
    for _, row in final_df.head(5).iterrows():
        target_pos = row['target_position']
        context_len = row['context_length']
        print(f"Word: {row['word']}, Position: {target_pos}/{context_len-1}, Expected: ~{window_size}")

if __name__ == "__main__":
    main()