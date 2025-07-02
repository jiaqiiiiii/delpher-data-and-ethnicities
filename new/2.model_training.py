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
import spacy
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import time

df = pd.read_csv('combined_contexts_1623_2025.csv')

print("=" * 60)
print("CSV FILE DIAGNOSIS")
print("=" * 60)
print(f"DataFrame shape: {df.shape}")
print(f"Available columns: {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head())

# Check if required columns exist
required_columns = ['context', 'year', 'word_group']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"\n❌ Missing required columns: {missing_columns}")
    print("Looking for similar column names...")
    
    for missing_col in missing_columns:
        similar_cols = [col for col in df.columns if missing_col.lower() in col.lower() or col.lower() in missing_col.lower()]
        if similar_cols:
            print(f"  Possible matches for '{missing_col}': {similar_cols}")
    
    print("\nPlease check your CSV structure and update column names accordingly.")
    exit(1)

# Continue with analysis if all required columns exist
print("✅ All required columns found. Proceeding with analysis...")

# Year visualization
if 'year' in df.columns:
    try:
        df0 = df.groupby(['year']).count()
        df0 = df0['file_path'] if 'file_path' in df.columns else df0.iloc[:, 0]
        
        ax = df0.plot(kind='bar', figsize=(20,6), color="indigo", fontsize=11)
        ax.set_title("How many files per year", fontsize=22)
        plt.show()
        print("✅ Year visualization completed")
    except Exception as e:
        print(f"⚠️  Year visualization failed: {e}")

# SpaCy processing
print("\n" + "=" * 60)
print("SPACY PROCESSING")
print("=" * 60)

try:
    sp = spacy.load("nl_core_news_sm")
    print("✅ SpaCy Dutch model loaded")
except Exception as e:
    print(f"❌ Failed to load SpaCy model: {e}")
    print("Please install the Dutch model: python -m spacy download nl_core_news_sm")
    exit(1)

# Ensure context column is string
df['context'] = df['context'].astype(str)
print(f"✅ Converted context column to string. Sample: {df['context'].iloc[0][:100]}...")

def process_text(text):
    try:
        doc = sp(text)
        return {
            'tokens': [word.text for word in doc],
            'lemmas': [word.lemma_ for word in doc],
            'doc': doc
        }
    except Exception as e:
        print(f"Error processing text: {e}")
        return {'tokens': [], 'lemmas': [], 'doc': None}

print("🔄 Starting SpaCy processing...")
try:
    # Process a small sample first to test
    sample_size = min(100, len(df))
    print(f"Testing with first {sample_size} rows...")
    
    sample_processed = df['context'].iloc[:sample_size].apply(process_text)
    print("✅ Sample processing successful")
    
    # Process full dataset
    print("🔄 Processing full dataset...")
    processed = df['context'].apply(process_text)
    
    df['Tokens'] = processed.apply(lambda x: x['tokens'])
    df['Lemmas'] = processed.apply(lambda x: x['lemmas'])
    
    print("✅ SpaCy processing completed")
    print(f"Sample tokens: {df['Tokens'].iloc[0][:5]}")
    print(f"Sample lemmas: {df['Lemmas'].iloc[0][:5]}")

except Exception as e:
    print(f"❌ SpaCy processing failed: {e}")
    exit(1)

# Check if Lemmas column was created successfully
if 'Lemmas' not in df.columns:
    print("❌ 'Lemmas' column not created. Cannot proceed.")
    exit(1)

print(f"✅ 'Lemmas' column created with {len(df)} rows")

# Text cleaning
print("\n🔄 Cleaning lemmas...")
try:
    df['Lemmas_clean'] = df['Lemmas'].apply(
        lambda lemmas: [t for t in lemmas if t not in '''!()-[]{};:\'",<>./?@#$%^&*_~|–—"'`''' and t.isalpha()]
    )
    print("✅ Text cleaning completed")
    print(f"Sample cleaned lemmas: {df['Lemmas_clean'].iloc[0][:5]}")
except Exception as e:
    print(f"❌ Text cleaning failed: {e}")
    exit(1)

# Continue with stopwords and rest of your analysis...
print("\n✅ Ready to continue with stopword removal and model training")

#I manually added these stopwords from what i have seen from the VOC data. 
old_stoplist = ['„','den','wij','t','ende','—','ten','wel','uijt','soo','de','van','en','in',
                                        'ten','uyt','ƒ','maer','daer','f',
            'dese', 'des','den', 'ter', 'aen', 'soude',
            'sullen', 'sal', 'haer', 'sij', 'hadde',
            'desen','ofte','se','sijn', 'onse',
            'sonder', 'soo', 'eenige', 'sijne',
            'oock', 'alsoo', 'naer', 'weder',
            'seer', 'ende', 'mede', 'dog', 'dogh',
            'konnen', 'off', 'connen', 'buyten',
            'daarvan', 'souden', 'nae', 'wesen',
            'waarvan', 'aldaer', 'tegens', 'dien','vande','alhier','welke','aldaar','alle','aande','deselve','noch',
               'lb', 'zullen', 'moeten', 'enen', 'd', 'ander','hadden', 'sijnde','zijnde', 'hebbende', 'daarmet', 'alleen',
                # Historical Dutch variants (16th-17th century)
        'ende', 'oft', 'maer', 'want', 'alsoo', 'doch', 'echter',
        'dewijl', 'overmits', 'alhoewel', 'schoon', 'hoewel',
        'sijn', 'syn', 'sijnde', 'synde', 'waer', 'waeren', 'ware', 'waren',
        'heeft', 'hadde', 'hebben', 'hebbende', 'gheweest',
        'werden', 'wordende', 'werdende', 'geworden',
        'sullen', 'souden', 'soude', 'sou', 'sal', 'sult',
        'moghen', 'mochte', 'conde', 'konde',
        'deser', 'dese', 'dit', 'dien', 'die', 'dat',
        'haer', 'hem', 'hun', 'sy', 'hij', 'hy',
        'oock', 'noch', 'alreede', 'alree', 'doen', 'doe', 'doen',
        'seer', 'soo', 'alsoo', 'daer', 'hier', 'waer',
        'somtijds', 'altoos', 'nimmermeer', 'wel', 'niet', 'en', 
        'den', 'der', 'des', 'dese', 'desen', 'deser', 'dit', 'dien', 'die',
        'haere', 'haeren', 'sijne', 'sine', 'hunne', 'hunnen','zyn', 'zynde', 'zyde','zy','zig', 'zyne',
        'aen', 'uyt', 'uijt', 'tusschen', 'jegens', 'ontrent', 'omtrent',
        'nevens', 'benevens', 'mitsgaders', 'beneffens',
        'alwaer', 'aldaer', 'alhier', 'daer', 'hier', 'ginds', 'elders',
        'wederom', 'andermael', 'eerstdaeghs', 'voortaen',
        'god', 'heer', 'heere', 'jesus', 'christus', 'heilige', 'kerk', 'kerck',
        'predikant', 'dominee', 'pastoor', 'pater', 'broeder', 'zuster',
        'koning', 'koninck', 'prins', 'graaf', 'heer', 'meester', 'mr',
        'edele', 'hoogheid', 'majesteit',
        'al', 'elk', 'elck', 'yder', 'ieder', 'alle', 'allen',
        'men', 'man', 'lieden', 'volck', 'volk',
        'veel', 'vele', 'weinig', 'weinigh', 'min', 'meer', 'meest',
        'eerst', 'eersten', 'tweede', 'derde', 'laatste', 'laetste', 'groot',
        'immers', 'evenwel', 'intusschen', 'ondertusschen', 'derhalven',
        'alzoo', 'deswegen', 'daarom', 'daardoor', 'daarvoor', 'daarna',
        'waarbij', 'waardoor', 'waarvoor', 'waarna', 'waarbij',
        'te', 'de', 'en', 'op', 'in', 'is', 'an', 'on', 'at', 'be', 'to',
        'of', 'or', 'my', 'me', 'we', 'he', 'it', 'as', 'so', 'no', 'if',
        'een', 'eene', 'twee', 'drie', 'vier', 'vijf', 'zes', 'zeven', 'acht', 'negen', 'tien',
        'elf', 'twaalf', 'dertien', 'veertien', 'vijftien', 'zestien', 'zeventien', 'achttien', 'negentien', 'twintig',
        'dertig', 'veertig', 'vijftig', 'zestig', 'zeventig', 'tachtig', 'negentig', 'honderd', 'duizend',
        'uur', 'uuren', 'januari', 'februari', 'maart', 'april', 'mei', 'juni', 'juli', 'augustus', 'september', 'oktober', 'november', 'december', 
        'maandag', 'dinsdag', 'woensdag', 'donderdag', 'vrijdag', 'zaterdag', 'zondag'] 


try:
    stopwords_set = set(stopwords.words('dutch') + old_stoplist)
    df["Lemmas_nostop"] = df['Lemmas'].apply(
        lambda lemmas: [t for t in lemmas if t not in stopwords_set]
    )
    print("✅ Stopword removal completed")
except Exception as e:
    print(f"❌ Stopword removal failed: {e}")

#  Create separate dataframes for each word_group
df1 = df[df['word_group'] == 'moor'].copy()
df2 = df[df['word_group'] == 'inboorling'].copy()

print(f"df1 (moor): {len(df1):,} rows")
print(f"df2 (inboorling): {len(df2):,} rows")

#C ount unique files per year for each word group
def plot_files_per_year(df_word, word_name, color='indigo'):
    # Count unique files per year
    files_per_year = df_word.groupby('year')['file_path'].nunique().sort_index()
    
    # Create plot
    ax = files_per_year.plot(kind='bar', figsize=(20, 6), color=color, fontsize=11)
    ax.set_title(f"How many files per year - {word_name.title()}", fontsize=22)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Number of Files", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return files_per_year

# Plot for each word group
print("Plotting for 'moor':")
moor_files = plot_files_per_year(df1, 'moor', 'indigo')

print("\nPlotting for 'inboorling':")
inboorling_files = plot_files_per_year(df2, 'inboorling', 'orange')

# 
def print_vocab(model, top_n = None):
  count = 0
  if top_n is not None:
    for index, word in enumerate(model.wv.index_to_key):
      count+= 1
      if count < top_n:
        print(f"word #{index}/{len(model.wv.index_to_key)} is {word}")
  else:
    for index, word in enumerate(model.wv.index_to_key):
      print(f"word #{index}/{len(model.wv.index_to_key)} is {word}")

#moor
#300 dimensions:
start = time.time()
cbow_w5_f1_300 = gensim.models.Word2Vec(df1['Lemmas_clean'], min_count=1, vector_size=300, window=5, sg=0)
end = time.time()
print("cbow_w5_f1_300 has taken", round(end - start), "seconds")
start = time.time()
cbow_w5_f0_300 = gensim.models.Word2Vec(df1['Lemmas_clean'], min_count=0, vector_size=300, window = 5, sg = 0)
end = time.time()
print("cbow_w5_f0_300 has taken", round(end - start), "seconds")
start = time.time()
sg_w5_f1_300 = gensim.models.Word2Vec(df1['Lemmas_clean'], min_count=1, vector_size=300, window = 5, sg = 1)
end = time.time()
print("sg_w5_f1_300 has taken", round(end - start), "seconds")
start = time.time()
sg_w5_f0_300 = gensim.models.Word2Vec(df1['Lemmas_clean'], min_count=0, vector_size=300, window = 5, sg = 1)
end = time.time()
print("sg_w5_f0_300 has taken", round(end - start), "seconds")
start = time.time()
cbow_w10_f1_300 = gensim.models.Word2Vec(df1['Lemmas_clean'], min_count=1, vector_size=300, window = 10, sg = 0)
end = time.time()
print("cbow_w10_f1_300 has taken", round(end - start), "seconds")
start = time.time()
cbow_w10_f0_300 = gensim.models.Word2Vec(df1['Lemmas_clean'], min_count=0, vector_size=300, window = 10, sg = 0)
end = time.time()
print("cbow_w10_f0_300 has taken", round(end - start), "seconds")
start = time.time()
sg_w10_f1_300 = gensim.models.Word2Vec(df1['Lemmas_clean'], min_count=1, vector_size=300, window = 10, sg = 1)
end = time.time()
print("sg_w10_f1_300 has taken", round(end - start), "seconds")
start = time.time()
sg_w10_f0_300 = gensim.models.Word2Vec(df1['Lemmas_clean'], min_count=0, vector_size=300, window = 10, sg = 1)
end = time.time()
print("sg_w10_f0_300 has taken", round(end - start), "seconds")
#1000 dimensions:
start = time.time()
cbow_w5_f1_1000 = gensim.models.Word2Vec(df1['Lemmas_clean'], min_count=1, vector_size=1000, window=5, sg=0)
end = time.time()
print("cbow_w5_f1_1000 has taken", round(end - start), "seconds")
start = time.time()
cbow_w5_f0_1000 = gensim.models.Word2Vec(df1['Lemmas_clean'], min_count=0, vector_size=1000, window = 5, sg = 0)
end = time.time()
print("cbow_w5_f0_1000 has taken", round(end - start), "seconds")
start = time.time()
sg_w5_f1_1000 = gensim.models.Word2Vec(df1['Lemmas_clean'], min_count=1, vector_size=1000, window = 5, sg = 1)
end = time.time()
print("sg_w5_f1_1000 has taken", round(end - start), "seconds")
start = time.time()
sg_w5_f0_1000 = gensim.models.Word2Vec(df1['Lemmas_clean'], min_count=0, vector_size=1000, window = 5, sg = 1)
end = time.time()
print("sg_w5_f0_1000 has taken", round(end - start), "seconds")
start = time.time()
cbow_w10_f1_1000 = gensim.models.Word2Vec(df1['Lemmas_clean'], min_count=1, vector_size=1000, window = 10, sg = 0)
end = time.time()
print("cbow_w10_f1_1000 has taken", round(end - start), "seconds")
start = time.time()
cbow_w10_f0_1000 = gensim.models.Word2Vec(df1['Lemmas_clean'], min_count=0, vector_size=1000, window = 10, sg = 0)
end = time.time()
print("cbow_w10_f0_1000 has taken", round(end - start), "seconds")
start = time.time()
sg_w10_f1_1000 = gensim.models.Word2Vec(df1['Lemmas_clean'], min_count=1, vector_size=1000, window = 10, sg = 1)
end = time.time()
print("sg_w10_f1_1000 has taken", round(end - start), "seconds")
start = time.time()
sg_w10_f0_1000 = gensim.models.Word2Vec(df1['Lemmas_clean'], min_count=0, vector_size=1000, window = 10, sg = 1)
end = time.time()
print("sg_w10_f0_1000 has taken", round(end - start), "seconds")

# inboorling
#300 dimensions:
start = time.time()
cbow_w5_f1_300_2 = gensim.models.Word2Vec(df2['Lemmas_clean'], min_count=1, vector_size=300, window=5, sg=0)
end = time.time()
print("cbow_w5_f1_300_2 has taken", round(end - start), "seconds")
start = time.time()
cbow_w5_f0_300_2 = gensim.models.Word2Vec(df2['Lemmas_clean'], min_count=0, vector_size=300, window = 5, sg = 0)
end = time.time()
print("cbow_w5_f0_300_2 has taken", round(end - start), "seconds")
start = time.time()
sg_w5_f1_300_2 = gensim.models.Word2Vec(df2['Lemmas_clean'], min_count=1, vector_size=300, window = 5, sg = 1)
end = time.time()
print("sg_w5_f1_300_2 has taken", round(end - start), "seconds")
start = time.time()
sg_w5_f0_300_2 = gensim.models.Word2Vec(df2['Lemmas_clean'], min_count=0, vector_size=300, window = 5, sg = 1)
end = time.time()
print("sg_w5_f0_300_2 has taken", round(end - start), "seconds")
start = time.time()
cbow_w10_f1_300_2 = gensim.models.Word2Vec(df2['Lemmas_clean'], min_count=1, vector_size=300, window = 10, sg = 0)
end = time.time()
print("cbow_w10_f1_300_2 has taken", round(end - start), "seconds")
start = time.time()
cbow_w10_f0_300_2 = gensim.models.Word2Vec(df2['Lemmas_clean'], min_count=0, vector_size=300, window = 10, sg = 0)
end = time.time()
print("cbow_w10_f0_300_2 has taken", round(end - start), "seconds")
start = time.time()
sg_w10_f1_300_2 = gensim.models.Word2Vec(df2['Lemmas_clean'], min_count=1, vector_size=300, window = 10, sg = 1)
end = time.time()
print("sg_w10_f1_300_2 has taken", round(end - start), "seconds")
start = time.time()
sg_w10_f0_300_2 = gensim.models.Word2Vec(df2['Lemmas_clean'], min_count=0, vector_size=300, window = 10, sg = 1)
end = time.time()
print("sg_w10_f0_300_2 has taken", round(end - start), "seconds")
#1000 dimensions:
start = time.time()
cbow_w5_f1_1000_2 = gensim.models.Word2Vec(df2['Lemmas_clean'], min_count=1, vector_size=1000, window=5, sg=0)
end = time.time()
print("cbow_w5_f1_1000_2 has taken", round(end - start), "seconds")
start = time.time()
cbow_w5_f0_1000_2 = gensim.models.Word2Vec(df2['Lemmas_clean'], min_count=0, vector_size=1000, window = 5, sg = 0)
end = time.time()
print("cbow_w5_f0_1000_2 has taken", round(end - start), "seconds")
start = time.time()
sg_w5_f1_1000_2 = gensim.models.Word2Vec(df2['Lemmas_clean'], min_count=1, vector_size=1000, window = 5, sg = 1)
end = time.time()
print("sg_w5_f1_1000_2 has taken", round(end - start), "seconds")
start = time.time()
sg_w5_f0_1000_2 = gensim.models.Word2Vec(df2['Lemmas_clean'], min_count=0, vector_size=1000, window = 5, sg = 1)
end = time.time()
print("sg_w5_f0_1000_2 has taken", round(end - start), "seconds")
start = time.time()
cbow_w10_f1_1000_2 = gensim.models.Word2Vec(df2['Lemmas_clean'], min_count=1, vector_size=1000, window = 10, sg = 0)
end = time.time()
print("cbow_w10_f1_1000_2 has taken", round(end - start), "seconds")
start = time.time()
cbow_w10_f0_1000_2 = gensim.models.Word2Vec(df2['Lemmas_clean'], min_count=0, vector_size=1000, window = 10, sg = 0)
end = time.time()
print("cbow_w10_f0_1000_2 has taken", round(end - start), "seconds")
start = time.time()
sg_w10_f1_1000_2 = gensim.models.Word2Vec(df2['Lemmas_clean'], min_count=1, vector_size=1000, window = 10, sg = 1)
end = time.time()
print("sg_w10_f1_1000_2 has taken", round(end - start), "seconds")
start = time.time()
sg_w10_f0_1000_2 = gensim.models.Word2Vec(df2['Lemmas_clean'], min_count=0, vector_size=1000, window = 10, sg = 1)
end = time.time()
print("sg_w10_f0_1000_2 has taken", round(end - start), "seconds")

#
# Define models folder - make sure it exists and is writable
models_folder = '/Users/zhujiaqi/Downloads/word2vec_models'  # Changed to a subfolder

# Create the directory if it doesn't exist
try:
    os.makedirs(models_folder, exist_ok=True)
    print(f"✅ Models folder created/verified: {models_folder}")
except Exception as e:
    print(f"❌ Failed to create models folder: {e}")
    # Fallback to current directory
    models_folder = './word2vec_models'
    os.makedirs(models_folder, exist_ok=True)
    print(f"✅ Using fallback folder: {models_folder}")

# Test write permissions
test_file = os.path.join(models_folder, 'test_write.txt')
try:
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    print("✅ Write permissions confirmed")
except Exception as e:
    print(f"❌ Write permission error: {e}")
    # Use current directory as final fallback
    models_folder = '.'
    print(f"✅ Using current directory: {models_folder}")

models = [
    # MOOR models in training order
    cbow_w5_f1_300, cbow_w5_f0_300, sg_w5_f1_300, sg_w5_f0_300, 
    cbow_w10_f1_300, cbow_w10_f0_300, sg_w10_f1_300, sg_w10_f0_300,
    cbow_w5_f1_1000,cbow_w5_f0_1000, sg_w5_f1_1000, sg_w5_f0_1000, cbow_w10_f1_1000, 
    cbow_w10_f0_1000, sg_w10_f1_1000, sg_w10_f0_1000,
    # INBOORLING models in training order  
    cbow_w5_f1_300_2, cbow_w5_f0_300_2, sg_w5_f1_300_2, sg_w5_f0_300_2,
    cbow_w10_f1_300_2, cbow_w10_f0_300_2, sg_w10_f1_300_2, sg_w10_f0_300_2,
    cbow_w5_f1_1000_2, cbow_w5_f0_1000_2, sg_w5_f1_1000_2, sg_w5_f0_1000_2, cbow_w10_f1_1000_2,
    cbow_w10_f0_1000_2, sg_w10_f1_1000_2, sg_w10_f0_1000_2
]

models_names = [
    # MOOR model names in same order as training
    'cbow_w5_f1_300', 'cbow_w5_f0_300', 'sg_w5_f1_300', 'sg_w5_f0_300', 
    'cbow_w10_f1_300', 'cbow_w10_f0_300', 'sg_w10_f1_300', 'sg_w10_f0_300',
    'cbow_w5_f1_1000','cbow_w5_f0_1000', 'sg_w5_f1_1000', 'sg_w5_f0_1000', 'cbow_w10_f1_1000', 
    'cbow_w10_f0_1000', 'sg_w10_f1_1000', 'sg_w10_f0_1000',
    # INBOORLING model names in same order as training
    'cbow_w5_f1_300_2', 'cbow_w5_f0_300_2', 'sg_w5_f1_300_2', 'sg_w5_f0_300_2',
    'cbow_w10_f1_300_2', 'cbow_w10_f0_300_2', 'sg_w10_f1_300_2', 'sg_w10_f0_300_2',
    'cbow_w5_f1_1000_2','cbow_w5_f0_1000_2', 'sg_w5_f1_1000_2', 'sg_w5_f0_1000_2', 'cbow_w10_f1_1000_2',
    'cbow_w10_f0_1000_2', 'sg_w10_f1_1000_2', 'sg_w10_f0_1000_2'
]

# Save models with error handling
successful_saves = 0
failed_saves = 0

for i in range(len(models)):
    model_name = models_names[i]
    model_path = os.path.join(models_folder, model_name)
    
    print(f"Saving model {i+1}/{len(models)}: {model_name}")
    
    try:
        models[i].save(model_path)
        print(f"  ✅ Successfully saved to {model_path}")
        successful_saves += 1
    except Exception as e:
        print(f"  ❌ Failed to save {model_name}: {e}")
        failed_saves += 1
        
        # Try alternative save method
        try:
            # Save to current directory as backup
            backup_path = f"./{model_name}_backup"
            models[i].save(backup_path)
            print(f"  ✅ Backup saved to {backup_path}")
            successful_saves += 1
        except Exception as e2:
            print(f"  ❌ Backup save also failed: {e2}")

print(f"\n📊 SAVE SUMMARY:")
print(f"  Successful saves: {successful_saves}")
print(f"  Failed saves: {failed_saves}")
print(f"  Models folder: {models_folder}")

if successful_saves > 0:
    print("✅ At least some models were saved successfully!")
else:
    print("❌ No models were saved. Check permissions and disk space.")

#
def calculate_mean_similarity(model, target_word, topn=10):
    """
    Calculate mean cosine similarity between target word and its top neighbors
    """
    try:
        if target_word not in model.wv.key_to_index:
            return None
        
        # Get top neighbors
        neighbors = model.wv.most_similar(target_word, topn=topn)
        
        if not neighbors:
            return None
        
        # Calculate cosine similarities
        target_vector = model.wv[target_word]
        similarities = []
        
        for neighbor, _ in neighbors:
            if neighbor in model.wv.key_to_index:
                neighbor_vector = model.wv[neighbor]
                similarity = 1 - cosine(target_vector, neighbor_vector)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else None
        
    except Exception as e:
        print(f"Error calculating similarity for {target_word}: {e}")
        return None

def grid_search_word2vec(df_data, target_word, word_group_name):
    """
    Perform grid search to find best Word2Vec model for a specific word
    """
    print(f"\n{'='*60}")
    print(f"GRID SEARCH FOR '{target_word.upper()}' ({word_group_name})")
    print(f"{'='*60}")
    
    # Check input data first
    print(f"Input data shape: {len(df_data)} sentences")
    if len(df_data) == 0:
        print("❌ No data provided for training")
        return None, None, pd.DataFrame()
    
    # Sample a few sentences to check format
    sample_sentences = df_data.iloc[:3].tolist()
    print(f"Sample sentences: {sample_sentences}")
    
    # Define parameter grid
    param_grid = {
        'sg': [0, 1],  # CBOW vs Skip-gram
        'vector_size': [300, 1000],
        'window': [5, 10],
        'min_count': [0, 1]
    }
    
    results = []
    best_score = -1
    best_model = None
    best_params = None
    total_attempts = 0
    successful_attempts = 0
    
    # Grid search
    for sg in param_grid['sg']:
        for vector_size in param_grid['vector_size']:
            for window in param_grid['window']:
                for min_count in param_grid['min_count']:
                    
                    total_attempts += 1
                    
                    # Create model name
                    algo_name = "sg" if sg == 1 else "cbow"
                    model_name = f"{algo_name}_w{window}_f{min_count}_{vector_size}"
                    
                    print(f"\nAttempt {total_attempts}: Training {model_name}...")
                    
                    start = time.time()
                    try:
                        # Train model
                        model = gensim.models.Word2Vec(
                            df_data, 
                            min_count=min_count, 
                            vector_size=vector_size, 
                            window=window, 
                            sg=sg,
                            epochs=5,  # Add epochs for better training
                            workers=4
                        )
                        
                        end = time.time()
                        training_time = round(end - start, 2)
                        
                        # Check if model was created successfully
                        if len(model.wv.index_to_key) == 0:
                            print(f"  ❌ Model has empty vocabulary")
                            continue
                        
                        print(f"  ✓ Model trained successfully")
                        print(f"  ✓ Vocabulary size: {len(model.wv.index_to_key):,}")
                        print(f"  ✓ Training time: {training_time}s")
                        
                        # Calculate mean similarity for target word
                        mean_sim = calculate_mean_similarity(model, target_word)
                        
                        if mean_sim is not None:
                            successful_attempts += 1
                            
                            # Store results
                            results.append({
                                'model_name': model_name,
                                'sg': sg,
                                'vector_size': vector_size,
                                'window': window,
                                'min_count': min_count,
                                'mean_similarity': mean_sim,
                                'training_time': training_time,
                                'vocab_size': len(model.wv.index_to_key),
                                'target_in_vocab': target_word in model.wv.key_to_index
                            })
                            
                            print(f"  ✓ Mean similarity: {mean_sim:.4f}")
                            
                            # Check if this is the best model
                            if mean_sim > best_score:
                                best_score = mean_sim
                                best_model = model
                                best_params = {
                                    'model_name': model_name,
                                    'sg': sg,
                                    'vector_size': vector_size,
                                    'window': window,
                                    'min_count': min_count
                                }
                                print(f"  🏆 NEW BEST MODEL! Score: {best_score:.4f}")
                        else:
                            print(f"  ❌ Target word '{target_word}' not found or error in similarity calculation")
                            results.append({
                                'model_name': model_name,
                                'sg': sg,
                                'vector_size': vector_size,
                                'window': window,
                                'min_count': min_count,
                                'mean_similarity': None,
                                'training_time': training_time,
                                'vocab_size': len(model.wv.index_to_key),
                                'target_in_vocab': False
                            })
                    
                    except Exception as e:
                        print(f"  ❌ Error training model: {e}")
                        results.append({
                            'model_name': model_name,
                            'sg': sg,
                            'vector_size': vector_size,
                            'window': window,
                            'min_count': min_count,
                            'mean_similarity': None,
                            'training_time': None,
                            'vocab_size': None,
                            'target_in_vocab': False
                        })
    
    # Print attempt summary
    print(f"\n📊 ATTEMPT SUMMARY:")
    print(f"  Total attempts: {total_attempts}")
    print(f"  Successful attempts: {successful_attempts}")
    print(f"  Results collected: {len(results)}")
    
    # Convert results to DataFrame with error handling
    if len(results) == 0:
        print("❌ No results to process - all training attempts failed")
        return None, None, pd.DataFrame()
    
    try:
        results_df = pd.DataFrame(results)
        print(f"✓ Created DataFrame with {len(results_df)} rows")
        
        # Check if we have any valid similarities
        valid_similarities = results_df['mean_similarity'].dropna()
        print(f"✓ Found {len(valid_similarities)} valid similarity scores")
        
        if len(valid_similarities) == 0:
            print("⚠️  No valid similarity scores found")
            # Still create the DataFrame but don't sort by similarity
            results_df.to_csv(f'grid_search_results_{word_group_name}.csv', index=False)
            return None, None, results_df
        
        # Sort by mean similarity (descending)
        results_df = results_df.sort_values('mean_similarity', ascending=False, na_last=True)
        
        # Save results
        results_df.to_csv(f'grid_search_results_{word_group_name}.csv', index=False)
        
    except Exception as e:
        print(f"❌ Error creating DataFrame: {e}")
        print(f"Results data: {results[:2]}")  # Show first 2 results for debugging
        return None, None, pd.DataFrame()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"GRID SEARCH SUMMARY FOR '{target_word.upper()}'")
    print(f"{'='*50}")
    
    if best_model is not None:
        print(f"🏆 BEST MODEL: {best_params['model_name']}")
        print(f"   Parameters: {best_params}")
        print(f"   Mean similarity: {best_score:.4f}")
        print(f"   Vocabulary size: {len(best_model.wv.index_to_key):,}")
    else:
        print("❌ No valid models found")
    
    # Show top 5 results if available
    valid_results = results_df[results_df['mean_similarity'].notna()]
    if len(valid_results) > 0:
        print(f"\n📊 TOP 5 MODELS:")
        print(f"{'Rank':<5} {'Model':<20} {'Mean Sim':<10} {'VecSize':<8} {'Window':<7} {'MinCount':<9} {'SG':<3}")
        print("-" * 70)
        
        for idx, (_, row) in enumerate(valid_results.head(5).iterrows()):
            print(f"{idx+1:<5} {row['model_name']:<20} {row['mean_similarity']:<10.4f} {row['vector_size']:<8} {row['window']:<7} {row['min_count']:<9} {row['sg']:<3}")
    else:
        print("❌ No valid results to display")
    
    return best_model, best_params, results_df

def find_best_models_separately(df1, df2, target_word1='moor', target_word2='inboorling'):
    """
    Find best model for each word independently - no comparison needed
    """
    print(f"\n{'='*80}")
    print("FINDING BEST MODELS FOR EACH WORD INDEPENDENTLY")
    print(f"{'='*80}")
    
    # Find best model for first word
    print(f"\n🎯 Finding best model for '{target_word1}'...")
    best_model1, best_params1, results1 = grid_search_word2vec(
        df1['Lemmas_clean'], target_word1, 'moor'
    )
    
    # Find best model for second word  
    print(f"\n🎯 Finding best model for '{target_word2}'...")
    best_model2, best_params2, results2 = grid_search_word2vec(
        df2['Lemmas_clean'], target_word2, 'inboorling'
    )
    
    # Just show the results - no comparison
    print(f"\n{'='*50}")
    print("BEST MODELS FOUND")
    print(f"{'='*50}")
    
    if best_model1:
        print(f"✅ Best model for '{target_word1}':")
        print(f"   Model: {best_params1['model_name']}")
        print(f"   Mean similarity: {calculate_mean_similarity(best_model1, target_word1):.4f}")
        print(f"   Parameters: {best_params1}")
        print(f"   Vocabulary: {len(best_model1.wv.index_to_key):,} words")
    else:
        print(f"❌ No valid model found for '{target_word1}'")
        
    if best_model2:
        print(f"\n✅ Best model for '{target_word2}':")
        print(f"   Model: {best_params2['model_name']}")
        print(f"   Mean similarity: {calculate_mean_similarity(best_model2, target_word2):.4f}")
        print(f"   Parameters: {best_params2}")
        print(f"   Vocabulary: {len(best_model2.wv.index_to_key):,} words")
    else:
        print(f"❌ No valid model found for '{target_word2}'")
    
    return {
        'moor': {'model': best_model1, 'params': best_params1, 'results': results1},
        'inboorling': {'model': best_model2, 'params': best_params2, 'results': results2}
    }

# Simplified usage - just find best models independently
best_models = find_best_models_separately(df1, df2)

# Access the best models
best_moor_model = best_models['moor']['model']
best_inboorling_model = best_models['inboorling']['model']

print(f"\n✅ Grid search complete!")
print(f"📁 Results saved to:")
print(f"   - grid_search_results_moor.csv")
print(f"   - grid_search_results_inboorling.csv")
print(f"\n🎯 Each word has its own optimal parameters - no need to match!")
