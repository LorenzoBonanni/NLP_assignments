import itertools
import random

from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer, FreqDist, NaiveBayesClassifier, download
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay, f1_score

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
def create_features(token_vector: list[list[str]], label: int):
    """
    Creates the dataset transforming each phrase into an occurrence dictionary based on the most frequent word,
    then assigning to each dictionary a label
    :param token_vector: a list containing tokenized sentences
    :param label: the label to assign to each data point
    :return: the dataset
    """

    label_vec = [label] * len(token_vector)
    feature_vec = [{f'token_{i}': token for i, token in enumerate(p)} for p in token_vector]
    return list(zip(feature_vec, label_vec))


# compute the dataset
labels = {"ENGLISH": 0, "NON-ENGLISH": 1}
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
print(f"ACCURACY: {round(accuracy_score(y_true, y_pred), 3)}")
print(f"PRECISION: {round(precision_score(y_true, y_pred), 3)}")
print(f"RECALL: {round(recall_score(y_true, y_pred), 3)}")
print(f"F1: {round(f1_score(y_true, y_pred), 3)}")
ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    display_labels=list(labels.keys()),
    xticks_rotation='vertical',
    cmap='Blues'
)
plt.tight_layout()
plt.savefig('img/confusion_matrix.png')
