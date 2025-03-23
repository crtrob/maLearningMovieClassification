# this file holds the HashingVectorizer used in the classifier
from sklearn.feature_extraction.text import HashingVectorizer
import pickle
import os
import re

# store current directory in variable
cur_dir = os.path.dirname(__file__)
# locate and store stopwords in localized variable, uses current directory variable
stop = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'stopwords.pkl'), 'rb'))

# replicate preprocessing & tokenizing method
def preproc_tokenize(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# replicate HashingVectorizer variable
vect = HashingVectorizer(decode_error = 'ignore', n_features = 2**21, preprocessor = None, 
                         tokenizer = preproc_tokenize)