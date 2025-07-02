import numpy as np
from gensim.models import Word2Vec
import pickle

# Load five models
# Replace these paths with your actual model paths
model1 = Word2Vec.load('word2vec_inboorling_period_1_sg_w10_f0_300.model')
model2 = Word2Vec.load('word2vec_inboorling_period_2_sg_w10_f0_300.model')
model3 = Word2Vec.load('word2vec_inboorling_period_3_sg_w10_f0_300.model')
model4 = Word2Vec.load('word2vec_inboorling_period_4_sg_w10_f0_300.model')
model5 = Word2Vec.load('word2vec_inboorling_period_5_sg_w10_f0_300.model')

def intersection_align_gensim_multiple(models, words=None):
    """
    Intersect multiple gensim word2vec models.
    Only the shared vocabulary between all models is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from all models).
    """
    
    # Get the vocab for each model
    vocabs = [set(m.wv.index_to_key) for m in models]
    
    # Find the common vocabulary across all models
    common_vocab = vocabs[0]
    for vocab in vocabs[1:]:
        common_vocab &= vocab
    
    if words: 
        common_vocab &= set(words)
    
    # Check if alignment is necessary
    alignment_needed = False
    for vocab in vocabs:
        if vocab - common_vocab:
            alignment_needed = True
            break
    
    if not alignment_needed:
        return models
    
    # Sort by frequency (summed across all models)
    common_vocab = list(common_vocab)
    common_vocab.sort(
        key=lambda w: sum(m.wv.get_vecattr(w, "count") for m in models), 
        reverse=True
    )
    
    print(f"Common vocabulary size: {len(common_vocab)}")
    
    # Align each model
    for i, m in enumerate(models):
        # Replace old vectors array with new one (with common vocab)
        indices = [m.wv.key_to_index[w] for w in common_vocab]
        old_arr = m.wv.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.vectors = new_arr
        
        # Replace old vocab dictionary and index mappings
        new_key_to_index = {}
        new_index_to_key = []
        for new_index, key in enumerate(common_vocab):
            new_key_to_index[key] = new_index
            new_index_to_key.append(key)
        m.wv.key_to_index = new_key_to_index
        m.wv.index_to_key = new_index_to_key
        
        print(f"Model {i+1}: {len(m.wv.key_to_index)} words, {len(m.wv.vectors)} vectors")
    
    return models

def intersection_align_gensim(m1, m2, words=None):
    """Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.index_to_key)
    vocab_m2 = set(m2.wv.index_to_key)

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1,m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w, "count"), reverse=True)
    # print(len(common_vocab))

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.key_to_index[w] for w in common_vocab]
        old_arr = m.wv.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        new_key_to_index = {}
        new_index_to_key = []
        for new_index, key in enumerate(common_vocab):
            new_key_to_index[key] = new_index
            new_index_to_key.append(key)
        m.wv.key_to_index = new_key_to_index
        m.wv.index_to_key = new_index_to_key
        
        print(len(m.wv.key_to_index), len(m.wv.vectors))
        
    return (m1,m2)

def procrustes_align_gensim(base_embed, other_embed):
    """
    Procrustes align two gensim word2vec models that already have aligned vocabularies.
    Assumes vocabularies are already intersected and aligned.
    
    Apply orthogonal Procrustes alignment to align other_embed to base_embed's coordinate system.
    """
    
    # get the (normalized) embedding matrices
    base_vecs = base_embed.wv.get_normed_vectors()
    other_vecs = other_embed.wv.get_normed_vectors()

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v) 
    # Replace original array with modified one, i.e. multiplying the embedding matrix by "ortho"
    other_embed.wv.vectors = (other_embed.wv.vectors).dot(ortho)    
    
    return other_embed

# First, align vocabularies across all five models
print("Aligning vocabularies across all five models...")
models = [model1, model2, model3, model4, model5]
aligned_models = intersection_align_gensim_multiple(models, words=None)

# Unpack the aligned models
model1, model2, model3, model4, model5 = aligned_models

# Apply Procrustes alignment using model1 as the base
print("\nApplying Procrustes alignment...")
print("Using model1 as base, aligning other models to it...")

# Align all other models to model1
model2_aligned = procrustes_align_gensim(model1, model2)
model3_aligned = procrustes_align_gensim(model1, model3)
model4_aligned = procrustes_align_gensim(model1, model4)
model5_aligned = procrustes_align_gensim(model1, model5)

print("Procrustes alignment completed!")

# Save the aligned models
print("\nSaving aligned models...")
model1.save('aligned_model1.model')
model2_aligned.save('aligned_model2.model')
model3_aligned.save('aligned_model3.model')
model4_aligned.save('aligned_model4.model')
model5_aligned.save('aligned_model5.model')

print("All aligned models saved successfully!")
print(f"Common vocabulary size: {len(model1.wv.key_to_index)}")

# Optional: Save as pickle files as well
with open('aligned_models.pkl', 'wb') as f:
    pickle.dump({
        'model1': model1,
        'model2': model2_aligned,
        'model3': model3_aligned,
        'model4': model4_aligned,
        'model5': model5_aligned
    }, f)

print("Models also saved as pickle file: aligned_models.pkl")