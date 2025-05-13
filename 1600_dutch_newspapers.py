import os
import glob
import re
import pandas as pd
from bs4 import BeautifulSoup


# Configuration
target_words = ["inboorling", "moor",]
base_dir = "/data/groups/trifecta/jiaqiz/downloaded_zip_delpher/kranten_pd_16xx"
window_size = 100
results = []


# Process each subdirectory matching kranten_pd_16* pattern
for sub_dir in glob.glob(os.path.join(base_dir, "16*")):
    for root, dirs, files in os.walk(sub_dir):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")


                # Extract year from directory structure
                year_match = re.search(r"/(\d{4})/", root)
                year = year_match.group(1) if year_match else "unknown"


                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        soup = BeautifulSoup(f, "xml")
                        text = soup.get_text(separator=' ', strip=True)


                        if not text.strip():
                            continue


                        words = text.split()
                        
                        # Find target words with context
                        for i, token in enumerate(words):
                            for word in target_words:
                                if token.lower() == word.lower():
                                    start = max(i - window_size, 0)
                                    end = i + window_size + 1
                                    context = " ".join(words[start:end])
                                    
                                    results.append({
                                        "word": word,
                                        "context": context,
                                        "year": year,
                                        "file_path": file_path
                                    })


                except Exception as e:
                    print(f"Error processing {file_path}: {e}")


# Create DataFrame and save results
df = pd.DataFrame(results)
print(f"Total contexts found: {len(df)}")
df.to_csv("contexts_by_word.csv", index=False, encoding="utf-8")
print("Results saved to contexts_by_word.csv")
