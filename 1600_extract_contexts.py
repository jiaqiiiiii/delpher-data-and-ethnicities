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

# Configuration
target_words = ["moor", "moren","moorin","inboorling", "inboorlingen"]
base_dir = "/data/groups/trifecta/jiaqiz/downloaded_zip_delpher/kranten_pd_16xx"
window_size = 50
similarity_threshold = 0.85  # Adjust as needed: higher = more duplicates kept

# Helper function to create a content signature
def get_signature(text):
    """Generate a normalized signature for deduplication"""
    # Remove multiple spaces and normalize whitespace
    text = re.sub(r'\s+', ' ', text.lower().strip())
    # Optionally: remove punctuation if needed
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Helper function to get MD5 hash of text for exact duplicate detection
def get_hash(text):
    """Generate MD5 hash for exact duplicate detection"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def find_target_words_in_text(text, year, file_path):
    """Find target words in text with proper context extraction"""
    results = []
    
    # Clean and tokenize the text
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    
    # Process each word
    for i, token in enumerate(words):
        clean_token = re.sub(r'[^\w]', '', token.lower())
        for target_word in target_words:
            # Check if the clean token matches our target word
            if clean_token == target_word.lower():
                # Calculate start and end positions with proper windowing
                start = max(i - window_size, 0)
                end = min(i + window_size + 1, len(words))
                
                # Extract surrounding context
                context = " ".join(words[start:end])
                
                # Store the result
                results.append({
                    "word": target_word,
                    "context": context,
                    "year": year,
                    "file_path": file_path,
                    "original_token": token,  # Keep the original token for reference
                    "signature": get_signature(context),  # Add signature for deduplication
                    "hash": get_hash(context)  # Add hash for exact duplicate detection
                })
    
    return results

def remove_similar_contexts(contexts_df, threshold=0.85):
    """Remove similar contexts using cosine similarity"""
    print("Removing similar contexts...")
    
    # If we have very few contexts, skip similarity check
    if len(contexts_df) <= 1:
        return contexts_df
    
    # Extract contexts for similarity comparison
    contexts = contexts_df['context'].tolist()
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words=['de', 'het', 'een', 'en', 'van', 'in', 'is'])
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
                # Keep the shorter one as it's likely more focused
                if len(contexts[i]) <= len(contexts[j]):
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
    for century in ["16xx"]:
        century_dir = base_dir.replace("16xx", century)
        
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
    
    # Remove exact duplicates first
    initial_count = len(df)
    df = df.drop_duplicates(subset=['hash'])
    print(f"Removed {initial_count - len(df)} exact duplicates")
    
    # Group by word and remove similar contexts within each group
    final_df = pd.DataFrame()
    for word, group in df.groupby('word'):
        print(f"Processing {len(group)} contexts for '{word}'...")
        filtered_group = remove_similar_contexts(group, similarity_threshold)
        final_df = pd.concat([final_df, filtered_group])
    
    # Reset index and drop unnecessary columns
    final_df = final_df.reset_index(drop=True)
    final_df = final_df.drop(['signature', 'hash'], axis=1)
    
    # Save results
    print(f"Total unique contexts found: {len(final_df)}")
    final_df.to_csv("unique_contexts_by_word.csv", index=False, encoding="utf-8")
    print("Results saved to unique_contexts_by_word.csv")
    
    # Save a summary by word and century
    summary = final_df.groupby(['word', final_df['year'].str[:2]]).size().reset_index(name='count')
    summary.columns = ['word', 'century', 'count']
    summary.to_csv("contexts_summary.csv", index=False)
    print("Summary saved to contexts_summary.csv")

if __name__ == "__main__":
    main()