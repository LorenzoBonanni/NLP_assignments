import itertools
import random

from nltk import download, WordNetLemmatizer, FreqDist
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

random.seed(1)
download('stopwords')
download('punkt')
download('wordnet')
download('omw-1.4')


################################################
#                                              #
#              DATA PREPROCESSING              #
#                                              #
################################################
def preprocess(language: str, text: str):
    """
    Preprocessing Pipeline to prepare the text for classification
    :param language: the language given as input
    :param text: the text to preprocess
    :return:
    """
    # split the text into sentences
    sentences = sent_tokenize(text, language)
    # tokenize each sentence
    tokenized_sentences = [word_tokenize(s, language) for s in sentences]
    # remove stop words
    stop_words = set(stopwords.words(language))
    filtered = [[word for word in s if word.casefold() not in stop_words] for s in tokenized_sentences]
    # remove punctuation, stem and lemmatize words
    stemmer = SnowballStemmer(language)
    lemmatizer = WordNetLemmatizer()
    final_token_vec = [
        [
            lemmatizer.lemmatize(stemmer.stem(word.lower()))
            for word in s if word.isalpha()
        ]
        for s in filtered
    ]
    return final_token_vec


############################
# ENGLISH DATA PROCESSING #
###########################
english1 = open('genesis/english-kjv.txt', encoding='utf-8').read()
english2 = open('genesis/english-web.txt', encoding='utf-8').read()
english1_token_vec = preprocess('english', english1)
english2_token_vec = preprocess('english', english2)
english_token_vec = english1_token_vec + english2_token_vec

###############################
# NON-ENGLISH DATA PROCESSING #
###############################
finnish = open('genesis/finnish.txt', encoding='latin-1').read()
french = open('genesis/french.txt', encoding='latin-1').read()
finnish_token_vec = preprocess('finnish', finnish)
french_token_vec = preprocess('french', french)
non_english_token_vec = french_token_vec + finnish_token_vec

########################
#   COMPUTE FEATURES   #
########################
# Flatten the token vectors
flat_english1 = list(itertools.chain(*english1_token_vec))
flat_english2 = list(itertools.chain(*english2_token_vec))
flat_finnish = list(itertools.chain(*finnish_token_vec))
flat_french = list(itertools.chain(*french_token_vec))
all_tokens = flat_finnish + flat_french + flat_english1 + flat_english2
N = 2000

def document_features(phrase):
    return {f'contains({word})': (word in set(phrase)) for word in most_common}

freq_dist = FreqDist(all_tokens)
most_common = map(lambda x: x[0], freq_dist.most_common(N))

# TODO tranform each phrase into a dictionary of features
# TODO do train and test
# TODO compute metrics (add F1)