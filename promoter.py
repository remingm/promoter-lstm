'''
Michael Remington
Promoter Identification with an LSTM

http://remingm.github.io/machine/learning/tensorflow/neural/networks/bioinformatics/2016/07/13/promoter-analysis.html

This code loads promoter sequences and labels, converts the sequences
to numbers via a dictionary, splits the data into randomly shuffled train
and test sets, trains and tunes an LSTM model with random hyper parameter sampling,
tests the model against a test sequence, and records the predicted probability for each
test sequence.

Only runs on TensorFlow 0.8.
'''

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import pandas

import tensorflow.contrib.skflow as skflow
import random
import csv

### Training data

train = pandas.read_csv('promoter/promoters.data', header=None)
X, y = train[2], train[0]
print(X.head())

Y_classes = len(train[0].unique())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
actual = y_test
print(y_test.head())

### Process vocabulary

MAX_DOCUMENT_LENGTH = 57

vocab_processor = skflow.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
X_train = np.array(list(vocab_processor.fit_transform(X_train)))
X_test = np.array(list(vocab_processor.transform(X_test)))
print(X_test)

n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)

### Process Lablels
cat_processor = skflow.preprocessing.text.VocabularyProcessor(1)
y_train = pandas.DataFrame(np.array(list(cat_processor.fit_transform(y_train))))
y_test = pandas.DataFrame(np.array(list(cat_processor.fit_transform(y_test))))
print(y_test.head())

### Models

EMBEDDING_SIZE = 1


# Customized function to transform batched X into embeddings
def input_op_fn(X):
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
                                                   embedding_size=EMBEDDING_SIZE, name='words')
    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = skflow.ops.split_squeeze(1, MAX_DOCUMENT_LENGTH, word_vectors)
    return word_list


# Random sampling for hyperparameter search
TUNE = True

if TUNE:
    NLAYERS = random.randint(1, 3)
    CELL = random.choice(['gru', 'lstm', 'rnn'])
    BATCH_SIZE = random.randrange(16, 200, 8)
    STEPS = 1000
    early_stopping_rounds = random.randrange(0, STEPS, 100)
    OPTIMIZER = random.choice(['SGD', 'Adam', 'Adagrad', 'Ftrl', 'RMSProp', ])
    LEARNING_RATE = random.uniform(0.001, 0.1)

else:
    NLAYERS = 2
    BATCH_SIZE = 128
    STEPS = 1000
    early_stopping_rounds = STEPS // 2
    CELL = 'lstm'
    OPTIMIZER = 'Adam'
    LEARNING_RATE = 0.035

# Add Hyper Parameters to Dict
hyperParameters = {"cell:": CELL, "hidden_layers": NLAYERS, "batch_size": BATCH_SIZE, "optimizer": OPTIMIZER,
                   "learning_rate": LEARNING_RATE, "Steps": STEPS, "early_stopping": early_stopping_rounds}
print(hyperParameters)

# We'll make a monitor so that we can implement early stopping based on our dev accuracy. This will prevent overfitting.
monitor = skflow.monitors.ValidationMonitor(val_X=X_test, val_y=y_test, n_classes=Y_classes,
                                            early_stopping_rounds=early_stopping_rounds)
# monitor = skflow.monitors.BaseMonitor(early_stopping_rounds=early_stopping_rounds)
# RNN model
classifier = skflow.TensorFlowRNNClassifier(rnn_size=EMBEDDING_SIZE,
                                            n_classes=Y_classes, cell_type=CELL, input_op_fn=input_op_fn,
                                            num_layers=NLAYERS, bidirectional=False, sequence_length=None,
                                            steps=STEPS, optimizer=OPTIMIZER, learning_rate=LEARNING_RATE,
                                            batch_size=BATCH_SIZE,
                                            continue_training=True)

# train  & predict on test set.
classifier.fit(X_train, y_train, logdir=None, monitor=monitor)
score = metrics.accuracy_score(y_test, classifier.predict(X_test))
print('Accuracy: {0:f}'.format(score))

# Add Accuracy to Dict
hyperParameters["Accuracy"] = score

# CSV file for promoter analysis
csvfile = open('promoterAnalysis' + str(score) + '.csv', 'wb')
csvwriter = csv.writer(csvfile)
# Headers
csvwriter.writerow(
    ['Sequence', 'Promoter', 'Predicted Probability of Promoter', 'A Fraction', 'T Fraction', 'C Fraction',
     'G Fraction', 'Contains common bacterial promoter sequence'])

# Get softmax probabilities
probs = classifier.predict_proba(X_test)
X_test = vocab_processor.reverse(X_test)
iter = 0
actual = actual.values
for i in X_test:

    print(i, probs[iter], actual[iter])

    # Write to csv
    containsSeq = False
    if 't a t a a t' in i or 't t g a c a' in i: containsSeq = True
    # Get nucleotide fractions
    t_count = 0.0
    a_count = 0.0
    c_count = 0.0
    g_count = 0.0
    for base in i.split():
        if base == 't': t_count += 1
        if base == 'a': a_count += 1
        if base == 'c': c_count += 1
        if base == 'g': g_count += 1

    total = t_count + a_count + c_count + g_count
    t_frac = float(t_count / total)
    a_frac = float(a_count / total)
    c_frac = float(c_count / total)
    g_frac = float(g_count / total)

    csvwriter.writerow([i, actual[iter], probs[iter][0], a_frac, t_frac, c_frac, g_frac, containsSeq])

    iter += 1

# Make Outfile
outFile = open("promoter_log.csv", "a")
outFile.write(str(hyperParameters))
outFile.write('\n')
