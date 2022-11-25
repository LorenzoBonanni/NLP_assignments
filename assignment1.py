import random

import matplotlib.pyplot as plt
from nltk import download
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import FrenchStemmer, EnglishStemmer, FinnishStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay

random.seed(1)
download('stopwords')
download('punkt')

################################################
#                                              #
#              DATA PREPROCESSING              #
#                                              #
################################################

############################
# ENGLISH DATA PROCESSING #
###########################
english1 = open('genesis/english-kjv.txt', encoding='utf-8').read()
english2 = open('genesis/english-web.txt', encoding='utf-8').read()
english_stop_words = set(stopwords.words("english"))

english1 = word_tokenize(english1, 'english')
english2 = word_tokenize(english2, 'english')

# Join Datasets
english = english1 + english2
# remove stop words
filtered_list_english = [word for word in english if word.casefold() not in english_stop_words]
# remove punctuation
list_english_no_punctuation = [word.lower() for word in filtered_list_english if word.isalpha()]
# stem each word
e_stemmer = EnglishStemmer()
list_english_stemmed = [{'token': e_stemmer.stem(word)} for word in list_english_no_punctuation]

english_labels = [0] * len(list_english_stemmed)
english_dataset = list(zip(list_english_stemmed, english_labels))

###############################
# NON-ENGLISH DATA PROCESSING #
###############################
finnish = open('genesis/finnish.txt', encoding='latin-1').read()
finnish_stop_words = set(stopwords.words("finnish"))
finnish = word_tokenize(finnish, 'finnish')
# remove stop words
filtered_list_finnish = [word for word in finnish if word.casefold() not in finnish_stop_words]
# remove punctuation
list_finnish_no_punctuation = [word.lower() for word in filtered_list_finnish if word.isalpha()]
# stem each word
fi_stemmer = FinnishStemmer()
list_finnish_stemmed = [{'token': fi_stemmer.stem(word)} for word in list_finnish_no_punctuation]

french = open('genesis/french.txt', encoding='latin-1').read()
french_stop_words = set(stopwords.words("french"))
french = word_tokenize(french, 'french')
# remove stop words
filtered_list_french = [word for word in french if word.casefold() not in french_stop_words]
# remove punctuation
list_french_no_punctuation = [word.lower() for word in filtered_list_french if word.isalpha()]
# stem each word
fr_stemmer = FrenchStemmer()
list_french_stemmed = [{'token': fr_stemmer.stem(word)} for word in list_french_no_punctuation]

non_english_tokens = list_french_stemmed + list_finnish_stemmed
non_english_labels = [1] * len(non_english_tokens)
non_english_dataset = list(zip(non_english_tokens, non_english_labels))

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

classifier = NaiveBayesClassifier.train(train_set)

################################################
#                                              #
#                   METRICS                    #
#                                              #
################################################
# Metrics: accuracy, precision, recall
labels = ["ENGLISH", "NON-ENGLISH"]
y_pred = [classifier.classify(feats) for (feats, label) in test_set]
y_true = [label for (feats, label) in test_set]
print("MODEL PERFORMANCE")
print(f"ACCURACY: {accuracy_score(y_true, y_pred)}")
print(f"PRECISION: {precision_score(y_true, y_pred)}")
print(f"RECALL: {recall_score(y_true, y_pred)}")
ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    display_labels=labels,
    xticks_rotation='vertical',
    cmap='Blues'
)
plt.tight_layout()
plt.savefig('img/confusion_matrix.png')