# NLP ASSIGNMENTS
## Assignment 1
### Assignment Description
The assignment consists in the development, in NLTK, OpenNLP, SketchEngine or GATE/Annie a Naïve Bayes Classifier able to detect a single class in one of the corpora available as attachments to the chosen package, by distinguishing ENGLISH against NON-ENGLISH. In particular the classifier has to be:

1. Trained on a split subset of the chosen corpus, by either using an existing partition between sample documents for training and for test or by using a random splitter among the available ones;

2. Devised as a pipeline of any chosen format, including the simplest version based on word2vec on a list of words obtained by one of the available lexical resources.

The test of the classifier shall give out the measures of accuracy, precision, recall on the obtained confusion matrix and WILL NOT BE EVALUATED ON THE LEVEL OF THE PERFORMANCES. In other terms, when the confusion is produced, then the value of the assignment will be good, independently of the percentage of false positive and negative results.

Deliver a short set of comments on the experience (do not deliver the entire code, but link it on a public repository like GitHub or the GATE Repo). Discuss: size of the corpus, size of the split training and test sets, performance indicators employed and their nature, employability of the classifier as a Probabilistic Language Model.
### Comments
For the first assignment I've chosen the [genesis corpus](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/genesis.zip) because it contains multiple languages and that's useful for the task.

Specifically I've chosen four corpora:
- english-kjv
- english-web
- finnish
- french

Although the corpus contains more languages I've chosen 2 non english corpora to maintain the dataset balanced because
from previous machine learning courses I've learned that imbalanced datasets might cause the model not to behave well.

Since there was only one document for each language I've chosen to divide each document into sentences and then do sentence classification.
In total the documents contains 7867 sentences which seems to be a pretty good dataset for this simple task.

The Train Test Split that I've chosen is respectively 70 and 30%, that is a common split in Machine Learning.
This results in 5506 sentences for Training and 2361 for Testing.

### Results
![conf_matrix](img/confusion_matrix.png "Confusion Matrix")

As we can see from the Confusion Matrix most of the tokens are correctly predicted

For measuring the performance of the classifier I've employed 4 Metrics:
- **Accuracy:** Indicates the number of correct predictions out of the total number of predictions
- **Precision:** Indicates what fraction of positive predictions are correct.
- **Recall**:  indicates what fraction of all positive instances does the classifier correctly identify as positive.
- **F1**: combines precision & recall into a single number

The results are:
```
ACCURACY: 0.987
PRECISION: 0.996
RECALL: 0.98
F1: 0.988
```
Those are really great results. The Naive Bayes classifier can be seen as a probabilistic language model, 
in the sense that it can be seen as a series of Unigram models one for each class (in this case 2).
Naive Bayes compute the joint probability of a sentence by multiplying the prior probability by the conditional probability of 
a token given a class (like unigram models). Then it predicts the class which has the highest probability.