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

# Check if processed data already exists
processed_file = 'combined_contexts_1623_2025_processed.csv'

if os.path.exists(processed_file):
    print("=" * 60)
    print("LOADING PROCESSED DATA")
    print("=" * 60)
    print(f"✅ Found processed file: {processed_file}")
    print("🔄 Loading processed data...")
    
    df = pd.read_csv(processed_file)
    
    # Convert string representations back to lists
    import ast
    print("🔄 Converting string columns back to lists...")
    try:
        df['Tokens'] = df['Tokens'].apply(ast.literal_eval)
        df['Lemmas'] = df['Lemmas'].apply(ast.literal_eval)
        df['Lemmas_clean'] = df['Lemmas_clean'].apply(ast.literal_eval)
        df['Lemmas_nostop'] = df['Lemmas_nostop'].apply(ast.literal_eval)
        print("✅ Successfully loaded processed data!")
    except Exception as e:
        print(f"❌ Error converting columns: {e}")
        print("🔄 Will reprocess the data...")
        df = pd.read_csv('combined_contexts_1623_2025.csv')
        processed_file_exists = False
    else:
        processed_file_exists = True
else:
    print("=" * 60)
    print("PROCESSING RAW DATA")
    print("=" * 60)
    df = pd.read_csv('combined_contexts_1623_2025.csv')
    processed_file_exists = False

if not processed_file_exists:
    print("=" * 60)
    print("CSV FILE DIAGNOSIS")
    print("=" * 60)
    print(f"DataFrame shape: {df.shape}")
    print(f"Available columns: {df.columns.tolist()}")

    # Check if required columns exist
    required_columns = ['context', 'year', 'word_group']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"\n❌ Missing required columns: {missing_columns}")
        exit(1)

    print("✅ All required columns found. Proceeding with analysis...")

    # SpaCy processing
    print("\n" + "=" * 60)
    print("SPACY PROCESSING")
    print("=" * 60)

    try:
        sp = spacy.load("nl_core_news_sm")
        print("✅ SpaCy Dutch model loaded")
    except Exception as e:
        print(f"❌ Failed to load SpaCy model: {e}")
        exit(1)

    # Ensure context column is string
    df['context'] = df['context'].astype(str)
    print(f"✅ Converted context column to string")

    def process_text(text):
        try:
            doc = sp(text)
            return {
                'tokens': [word.text for word in doc],
                'lemmas': [word.lemma_ for word in doc],
                'doc': doc
            }
        except Exception as e:
            return {'tokens': [], 'lemmas': [], 'doc': None}

    print("🔄 Starting SpaCy processing...")
    try:
        # Process full dataset
        processed = df['context'].apply(process_text)
        df['Tokens'] = processed.apply(lambda x: x['tokens'])
        df['Lemmas'] = processed.apply(lambda x: x['lemmas'])
        print("✅ SpaCy processing completed")

    except Exception as e:
        print(f"❌ SpaCy processing failed: {e}")
        exit(1)

    # Text cleaning
    print("\n🔄 Cleaning lemmas...")
    try:
        df['Lemmas_clean'] = df['Lemmas'].apply(
            lambda lemmas: [t for t in lemmas if t not in '''!()-[]{};:\'",<>./?@#$%^&*_~|–—"'`''' and t.isalpha()]
        )
        print("✅ Text cleaning completed")
    except Exception as e:
        print(f"❌ Text cleaning failed: {e}")
        exit(1)

    # Stopwords
    old_stoplist = ['„','den','wij','t','ende','—','ten','wel','uijt','soo','de','van','en','in',
                    'ten','uyt','ƒ','maer','daer','f','dese', 'des','den', 'ter', 'aen', 'soude',
                    'sullen', 'sal', 'haer', 'sij', 'hadde','desen','ofte','se','sijn', 'onse',
                    'sonder', 'soo', 'eenige', 'sijne','oock', 'alsoo', 'naer', 'weder',
                    'seer', 'ende', 'mede', 'dog', 'dogh','konnen', 'off', 'connen', 'buyten',
                    'daarvan', 'souden', 'nae', 'wesen','waarvan', 'aldaer', 'tegens', 'dien','vande','alhier','welke','aldaar','alle','aande','deselve','noch',
                    'lb', 'zullen', 'moeten', 'enen', 'd', 'ander','hadden', 'sijnde','zijnde', 'hebbende', 'daarmet', 'alleen',
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

    # Save processed data
    print("\n🔄 Saving processed data...")
    try:
        df.to_csv(processed_file, index=False)
        print(f"✅ Processed data saved to: {processed_file}")
        print(f"   File size: {os.path.getsize(processed_file) / (1024*1024):.1f} MB")
    except Exception as e:
        print(f"❌ Failed to save processed data: {e}")

else:
    print(f"DataFrame shape: {df.shape}")
    print(f"Available columns: {df.columns.tolist()}")

# Create separate dataframes for each word_group
df1 = df[df['word_group'] == 'moor'].copy()
df2 = df[df['word_group'] == 'inboorling'].copy()

print(f"df1 (moor): {len(df1):,} rows")
print(f"df2 (inboorling): {len(df2):,} rows")

# ==================== LOAD PRE-TRAINED MODELS ====================

print("\n" + "=" * 80)
print("LOADING PRE-TRAINED MODELS")
print("=" * 80)

# Define models folder
models_folder = 'word2vec_models'  
if not os.path.exists(models_folder):
    models_folder = './word2vec_models'
if not os.path.exists(models_folder):
    models_folder = '.'

print(f"Looking for models in: {models_folder}")

# Model names that should exist
models_names = [
    # MOOR model names
    'cbow_w5_f1_300', 'cbow_w5_f0_300', 'sg_w5_f1_300', 'sg_w5_f0_300', 
    'cbow_w10_f1_300', 'cbow_w10_f0_300', 'sg_w10_f1_300', 'sg_w10_f0_300',
    'cbow_w5_f1_1000', 'cbow_w5_f0_1000', 'sg_w5_f1_1000', 'sg_w5_f0_1000', 'cbow_w10_f1_1000', 
    'cbow_w10_f0_1000', 'sg_w10_f1_1000', 'sg_w10_f0_1000',
    # INBOORLING model names
    'cbow_w5_f1_300_2', 'cbow_w5_f0_300_2', 'sg_w5_f1_300_2', 'sg_w5_f0_300_2',
    'cbow_w10_f1_300_2', 'cbow_w10_f0_300_2', 'sg_w10_f1_300_2', 'sg_w10_f0_300_2',
    'cbow_w5_f1_1000_2', 'cbow_w5_f0_1000_2', 'sg_w5_f1_1000_2', 'sg_w5_f0_1000_2', 'cbow_w10_f1_1000_2',
    'cbow_w10_f0_1000_2', 'sg_w10_f1_1000_2', 'sg_w10_f0_1000_2'
]

# Load models
loaded_models = {}
loading_summary = {'successful': 0, 'failed': 0, 'failed_models': []}

print(f"Attempting to load {len(models_names)} models...")

for model_name in models_names:
    model_path = os.path.join(models_folder, model_name)
    backup_path = f"./{model_name}_backup"
    
    # Try main path first, then backup
    for path_to_try in [model_path, backup_path]:
        try:
            if os.path.exists(path_to_try):
                print(f"Loading {model_name} from {path_to_try}...")
                model = gensim.models.Word2Vec.load(path_to_try)
                loaded_models[model_name] = model
                loading_summary['successful'] += 1
                print(f"  ✅ Loaded successfully - Vocab size: {len(model.wv.index_to_key):,}")
                break
        except Exception as e:
            print(f"  ❌ Failed to load from {path_to_try}: {e}")
    
    if model_name not in loaded_models:
        loading_summary['failed'] += 1
        loading_summary['failed_models'].append(model_name)

print(f"\n📊 MODEL LOADING SUMMARY:")
print(f"  Successfully loaded: {loading_summary['successful']}")
print(f"  Failed to load: {loading_summary['failed']}")
if loading_summary['failed_models']:
    print(f"  Failed models: {loading_summary['failed_models']}")

# Separate loaded models by word group
moor_models = {name: model for name, model in loaded_models.items() if not name.endswith('_2')}
inboorling_models = {name: model for name, model in loaded_models.items() if name.endswith('_2')}

print(f"\n📋 LOADED MODELS:")
print(f"  Moor models: {len(moor_models)}")
print(f"  Inboorling models: {len(inboorling_models)}")

# ==================== ANALYSIS FUNCTIONS ====================

def calculate_mean_similarity(model, target_word, topn=10):
    """Calculate mean cosine similarity between target word and its top neighbors"""
    try:
        if target_word not in model.wv.key_to_index:
            return None
        
        neighbors = model.wv.most_similar(target_word, topn=topn)
        if not neighbors:
            return None
        
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

def analyze_loaded_models(models_dict, target_word, word_group_name):
    """Analyze loaded models instead of training new ones"""
    print(f"\n{'='*60}")
    print(f"ANALYZING LOADED MODELS FOR '{target_word.upper()}' ({word_group_name})")
    print(f"{'='*60}")
    
    if not models_dict:
        print("❌ No models loaded for analysis")
        return None, None, pd.DataFrame()
    
    results = []
    best_score = -1
    best_model = None
    best_params = None
    
    print(f"Analyzing {len(models_dict)} loaded models...")
    
    for model_name, model in models_dict.items():
        print(f"\nAnalyzing model: {model_name}")
        
        # Parse model parameters from name
        parts = model_name.replace('_2', '').split('_')
        try:
            sg = 1 if parts[0] == 'sg' else 0
            window = int(parts[1][1:])  # Extract number from 'w5' or 'w10'
            min_count = int(parts[2][1:])  # Extract number from 'f0' or 'f1'
            vector_size = int(parts[3])  # Extract dimension
        except:
            print(f"  ❌ Could not parse parameters from {model_name}")
            continue
        
        try:
            vocab_size = len(model.wv.index_to_key)
            target_in_vocab = target_word in model.wv.key_to_index
            
            print(f"  ✓ Vocabulary size: {vocab_size:,}")
            print(f"  ✓ Target word in vocab: {target_in_vocab}")
            
            # Calculate mean similarity
            mean_sim = calculate_mean_similarity(model, target_word)
            
            if mean_sim is not None:
                results.append({
                    'model_name': model_name,
                    'sg': sg,
                    'vector_size': vector_size,
                    'window': window,
                    'min_count': min_count,
                    'mean_similarity': mean_sim,
                    'vocab_size': vocab_size,
                    'target_in_vocab': target_in_vocab
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
                    'vocab_size': vocab_size,
                    'target_in_vocab': target_in_vocab
                })
                
        except Exception as e:
            print(f"  ❌ Error analyzing {model_name}: {e}")
            continue
    
    # Convert results to DataFrame
    if not results:
        print("❌ No valid results from model analysis")
        return None, None, pd.DataFrame()
    
    try:
        results_df = pd.DataFrame(results)
        
        # Sort by mean similarity - FIXED: Use compatible sorting method
        valid_similarities = results_df['mean_similarity'].dropna()
        if len(valid_similarities) > 0:
            # Use sort_values without na_last parameter for older pandas versions
            try:
                results_df = results_df.sort_values('mean_similarity', ascending=False, na_last=True)
            except TypeError:
                # Fallback for older pandas versions
                results_df = results_df.sort_values('mean_similarity', ascending=False)
                # Move NaN values to the end manually
                nan_mask = results_df['mean_similarity'].isna()
                results_df = pd.concat([results_df[~nan_mask], results_df[nan_mask]])
        
        # Save results
        results_df.to_csv(f'loaded_models_analysis_{word_group_name}.csv', index=False)
        print(f"✅ Results saved to 'loaded_models_analysis_{word_group_name}.csv'")
        
    except Exception as e:
        print(f"❌ Error creating results DataFrame: {e}")
        return None, None, pd.DataFrame()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"ANALYSIS SUMMARY FOR '{target_word.upper()}'")
    print(f"{'='*50}")
    
    if best_model is not None:
        print(f"🏆 BEST MODEL: {best_params['model_name']}")
        print(f"   Parameters: {best_params}")
        print(f"   Mean similarity: {best_score:.4f}")
        print(f"   Vocabulary size: {len(best_model.wv.index_to_key):,}")
    else:
        print("❌ No valid models found")
    
    # Show top 5 results
    valid_results = results_df[results_df['mean_similarity'].notna()]
    if len(valid_results) > 0:
        print(f"\n📊 TOP 5 MODELS:")
        print(f"{'Rank':<5} {'Model':<20} {'Mean Sim':<10} {'VecSize':<8} {'Window':<7} {'MinCount':<9} {'SG':<3}")
        print("-" * 70)
        
        for idx, (_, row) in enumerate(valid_results.head(5).iterrows()):
            print(f"{idx+1:<5} {row['model_name']:<20} {row['mean_similarity']:<10.4f} {row['vector_size']:<8} {row['window']:<7} {row['min_count']:<9} {row['sg']:<3}")
    
    return best_model, best_params, results_df

# ==================== RUN ANALYSIS ON LOADED MODELS ====================

print("\n" + "=" * 80)
print("ANALYZING PRE-TRAINED MODELS")
print("=" * 80)

# Analyze models for each word
best_models = {}

# Analyze moor models
if moor_models:
    print(f"\n🎯 Analyzing models for 'moor'...")
    best_model_moor, best_params_moor, results_moor = analyze_loaded_models(
        moor_models, 'moor', 'moor'
    )
    best_models['moor'] = {
        'model': best_model_moor, 
        'params': best_params_moor, 
        'results': results_moor
    }
else:
    print("❌ No moor models loaded")
    best_models['moor'] = {'model': None, 'params': None, 'results': pd.DataFrame()}

# Analyze inboorling models
if inboorling_models:
    print(f"\n🎯 Analyzing models for 'inboorling'...")
    best_model_inboorling, best_params_inboorling, results_inboorling = analyze_loaded_models(
        inboorling_models, 'inboorling', 'inboorling'
    )
    best_models['inboorling'] = {
        'model': best_model_inboorling, 
        'params': best_params_inboorling, 
        'results': results_inboorling
    }
else:
    print("❌ No inboorling models loaded")
    best_models['inboorling'] = {'model': None, 'params': None, 'results': pd.DataFrame()}

# ==================== FINAL SUMMARY ====================

print("\n" + "=" * 80)
print("FINAL ANALYSIS SUMMARY")
print("=" * 80)

for word in ['moor', 'inboorling']:
    if best_models[word]['model'] is not None:
        model = best_models[word]['model']
        params = best_models[word]['params']
        
        print(f"\n✅ Best model for '{word}':")
        print(f"   Model: {params['model_name']}")
        print(f"   Mean similarity: {calculate_mean_similarity(model, word):.4f}")
        print(f"   Parameters: {params}")
        print(f"   Vocabulary: {len(model.wv.index_to_key):,} words")
    else:
        print(f"\n❌ No valid model found for '{word}'")

print(f"\n📁 OUTPUT FILES:")
output_files = []
for word in ['moor', 'inboorling']:
    if not best_models[word]['results'].empty:
        output_files.append(f'loaded_models_analysis_{word}.csv')

for f in output_files:
    print(f"   - {f}")

print(f"   - {processed_file} (processed SpaCy data)")

print(f"\n✅ ANALYSIS COMPLETE! No model training required - used pre-trained models!")
print("=" * 80)