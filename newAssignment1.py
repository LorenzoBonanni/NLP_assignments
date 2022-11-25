import itertools
import random

from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer, FreqDist, NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay, f1_score

random.seed(1)


# download('stopwords')
# download('punkt')
# download('wordnet')
# download('omw-1.4')


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
def create_features(token_vector: list[str], label: int):
    """
    Creates the dataset transforming each phrase into an occurrence dictionary based on the most frequent word,
    then assigning to each dictionary a label
    :param token_vector: a list containing tokenized sentences
    :param label: the label to assign to each data point
    :return: the dataset
    """
    def vec_to_feature(phrase):
        return {f'contains({word})': (word in set(phrase)) for word in most_common}

    label_vec = [label] * len(token_vector)
    feature_vec = [vec_to_feature(p) for p in token_vector]
    return list(zip(feature_vec, label_vec))


# Flatten the token vectors
flat_english1 = list(itertools.chain(*english1_token_vec))
flat_english2 = list(itertools.chain(*english2_token_vec))
flat_finnish = list(itertools.chain(*finnish_token_vec))
flat_french = list(itertools.chain(*french_token_vec))
all_tokens = flat_finnish + flat_french + flat_english1 + flat_english2
N = 2000
labels = {"ENGLISH": 0, "NON-ENGLISH": 1}
# compute most common words
freq_dist = FreqDist(all_tokens)
most_common = list(freq_dist)[:N]
# compute the dataset
english_dataset = create_features(english_token_vec, labels["ENGLISH"])
non_english_dataset = create_features(non_english_token_vec, labels["NON-ENGLISH"])

################################################
#                                              #
#                   CLASSIFY                   #
#                                              #
################################################
complete_dataset = english_dataset + non_english_dataset
random.shuffle(complete_dataset)

# Split Train and Test Set 70% Training and 30% for Test
train_split = int(0.7 * len(complete_dataset))
train_set, test_set = complete_dataset[:train_split], complete_dataset[train_split:]

print("TRAINING")
classifier = NaiveBayesClassifier.train(train_set)

################################################
#                                              #
#                   METRICS                    #
#                                              #
################################################
# Metrics: accuracy, precision, recall, f1
print("TESTING")
y_pred = [classifier.classify(feats) for (feats, label) in test_set]
y_true = [label for (feats, label) in test_set]
print("MODEL PERFORMANCE")
print(f"ACCURACY: {accuracy_score(y_true, y_pred)}")
print(f"PRECISION: {precision_score(y_true, y_pred)}")
print(f"RECALL: {recall_score(y_true, y_pred)}")
print(f"F1: {f1_score(y_true, y_pred)}")
ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    display_labels=list(labels.keys()),
    xticks_rotation='vertical',
    cmap='Blues'
)
plt.tight_layout()
plt.savefig('img/confusion_matrix.png')
