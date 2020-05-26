import numpy as np
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from fuzzywuzzy import fuzz
from textblob import TextBlob, Word
import readability
# from keras.preprocessing.text import text_to_word_sequence
# from gensim.models import KeyedVectors
# import dill
# import nltk
# # nltk.download('punkt')
# # nltk.download('wordnet')


def generate_features(df):
    # Calculate words_diff_q1_q2 = difference of the word counts between question1 and question2
    df['words_diff_q1_q2'] = abs(df['question1'].apply(lambda row: len(row.split(" "))) - df['question2'].apply(lambda row: len(row.split(" "))))
    # Calculate word_common = count of unique words in question1 and question2
    df['word_common'] = df.apply(normalized_word_common, axis=1)
    # Calculate word_total = total number of words in Question 1 + total number of words in Question 2
    df['word_total'] = df.apply(normalized_word_total, axis=1)
    # Calculate word_share = word_common / word_total
    df['word_share'] = df.apply(normalized_word_share, axis=1)
    df['cosine_distance'] = df.apply(get_cosine_distance, axis=1)
    df['cityblock_distance'] = df.apply(get_cityblock_distance, axis=1)
    df['jaccard_distance'] = df.apply(get_jaccard_distance, axis=1)
    df['canberra_distance'] = df.apply(get_canberra_distance, axis=1)
    df['euclidean_distance'] = df.apply(get_euclidean_distance, axis=1)
    df['minkowski_distance'] = df.apply(get_minkowski_distance, axis=1)
    df['braycurtis_distance'] = df.apply(get_braycurtis_distance, axis=1)
    #
    df['fuzz_qratio'] = df.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
    df['fuzz_WRatio'] = df.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
    df['fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
    df['fuzz_partial_token_set_ratio'] = df.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    df['fuzz_partial_token_sort_ratio'] = df.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
    df['fuzz_token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    df['fuzz_token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
    #
    df['common_bigrams'] = df.apply(get_common_bigrams_count, axis=1)
    df['common_trigrams'] = df.apply(get_common_trigrams_count, axis=1)
    #
    df['q1_readability_score'] = df.apply(get_q1_readability_score, axis=1)
    df['q2_readability_score'] = df.apply(get_q2_readability_score, axis=1)
    #
    df['word_2_vec_diff'] = 0.5#df.apply(get_word2vec_similarity_score, axis=1)

    return df

def normalized_word_common(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return len(w1 & w2)

def normalized_word_total(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return (len(w1) + len(w2))

def normalized_word_share(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return len(w1 & w2) / (len(w1) + len(w2))

def get_cosine_distance(row):
    q1_q2_vectors = get_q1_q2_vectors(row)
    return cosine(q1_q2_vectors[0], q1_q2_vectors[1])

def get_cityblock_distance(row):
    q1_q2_vectors = get_q1_q2_vectors(row)
    return cityblock(q1_q2_vectors[0], q1_q2_vectors[1])

def get_jaccard_distance(row):
    q1_q2_vectors = get_q1_q2_vectors(row)
    return jaccard(q1_q2_vectors[0], q1_q2_vectors[1])

def get_canberra_distance(row):
    q1_q2_vectors = get_q1_q2_vectors(row)
    return canberra(q1_q2_vectors[0], q1_q2_vectors[1])

def get_euclidean_distance(row):
    q1_q2_vectors = get_q1_q2_vectors(row)
    return euclidean(q1_q2_vectors[0], q1_q2_vectors[1])

def get_minkowski_distance(row):
    q1_q2_vectors = get_q1_q2_vectors(row)
    return minkowski(q1_q2_vectors[0], q1_q2_vectors[1])

def get_braycurtis_distance(row):
    q1_q2_vectors = get_q1_q2_vectors(row)
    return braycurtis(q1_q2_vectors[0], q1_q2_vectors[1])

# Function to return the vectors for q1 and q2
def get_q1_q2_vectors(row):
    if row['id'] not in q1_q2_vectors:
        vocab = set()
        q1_list = []
        q1_list.append(row['question1'].replace("\n", " "))
        vocab = vocab.union(set(convert(q1_list)))

        q2_list = []
        q2_list.append(row['question2'].replace("\n", " "))
        vocab = vocab.union(set(convert(q2_list)))

        # integer encode sequences of words
        tokenizer = Tokenizer(filters=[])
        tokenizer.fit_on_texts(vocab)

        q1_seq = tokenizer.texts_to_sequences(q1_list)
        q2_seq = tokenizer.texts_to_sequences(q2_list)

        PAD_LENGTH = len(q1_seq[0]) + len(q2_seq[0])
        q1_seq_padded = pad_sequences(q1_seq, maxlen=PAD_LENGTH)
        q2_seq_padded = pad_sequences(q2_seq, maxlen=PAD_LENGTH)

        q1_seq_sorted = np.sort(q1_seq_padded[0])
        q2_seq_sorted = np.sort(q2_seq_padded[0])

        q1_q2_list = []
        q1_q2_list.append(q1_seq_sorted)
        q1_q2_list.append(q2_seq_sorted)

        q1_q2_vectors[row['id']] = q1_q2_list

    return q1_q2_vectors[row['id']][0], q1_q2_vectors[row['id']][1]

# Vectorizing question1 and question2 and doing distance matrix computations
def convert(lst):
    return (lst[0].split())

q1_q2_vectors = {}

# Function to generate bigrams for every q1 and q2 and count the common bigrams.
def get_common_bigrams_count(row):
  q1_bigrams = TextBlob(row['question1'].replace("\n", " ")).ngrams(2)
  q2_bigrams = TextBlob(row['question2'].replace("\n", " ")).ngrams(2)

  common=[]
  for bigrams in q1_bigrams:
      if bigrams in q2_bigrams:
        common.append(bigrams)

  return len(common)

# Function to generate trigrams for every q1 and q2 and count the common trigrams.
def get_common_trigrams_count(row):
  q1_trigrams = TextBlob(row['question1'].replace("\n", " ")).ngrams(3)
  q2_trigrams = TextBlob(row['question2'].replace("\n", " ")).ngrams(3)

  common=[]
  for trigrams in q1_trigrams:
      if trigrams in q2_trigrams:
        common.append(trigrams)

  return len(common)

def get_q1_readability_score(row):
  text = row['question1']
  try:
    results = readability.getmeasures(text, lang='en')
  except:
    return 0.00
  return results['readability grades']['FleschReadingEase']

def get_q2_readability_score(row):
  text = row['question2']
  try:
    results = readability.getmeasures(text, lang='en')
  except:
    return 0.00
  return results['readability grades']['FleschReadingEase']

