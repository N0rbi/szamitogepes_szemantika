# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 12:57:21 2018

@author: norbi
"""

import pandas as pd
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten
from keras.callbacks import Callback
import matplotlib.pyplot as plt

# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
embedding_file = open('./GoogleNews-vectors-negative300_lite.txt')
embedding_file.readline()
embedding_indices = dict()
for line in embedding_file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype=np.float32)
    embedding_indices[word] = coefs
embedding_file.close()

def get_model(e):
    model = Sequential()
    model.add(e)
    model.add(Dense(64, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['acc'])

    print(model.summary())
    return model


class EpochHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('acc'))
        return


accs = list()
losses = list()

dataset_base = "lexical_entailment"
datasets = ["baroni2012", "bless2011", "kotlerman2010", "levy2014", "turney2014"]
test = "data_lex_test.tsv"
train = "data_lex_train.tsv"
val = "data_lex_val.tsv"

for chosen_dataset in range(len(datasets)):
    glob = pd.read_csv(
        os.path.join(dataset_base, datasets[chosen_dataset], 'data.tsv')
        , sep='\t', header=None)

    glob_text_np = np.array(glob.loc[:, 0:1]).flatten()
    glob_text_uniq = list(set(glob_text_np))
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(glob_text_uniq)
    vocab_size = len(tokenizer.word_index) + 1

    ###
    # Get the binary into readable format
    ###
    # https://stackoverflow.com/questions/27324292/convert-word2vec-bin-file-to-text
    # from gensim.models.keyedvectors import KeyedVectors
    # model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    # model.save_word2vec_format('./GoogleNews-vectors-negative300.txt', binary=False)


    ###
    # Load weights into keras readable format
    ###
    # https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    num_words_found = 0
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embedding_indices.get(word)
        if embedding_vector is not None:
            num_words_found+=1
            embedding_matrix[i] = embedding_vector

    print("%d words found out of %d in the %d item long embedding corpus."%
          (num_words_found, vocab_size, len(embedding_indices)))
    ###
    # Setting up the model
    ###

    embedding1 = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=2, trainable=False)
    model1 = get_model(embedding1)

    embedding2 = Embedding(vocab_size, 300, input_length=2, trainable=True)
    model2 = get_model(embedding2)

    ###
    # Train the model
    ###

    train_glob = pd.read_csv(
        os.path.join(dataset_base, datasets[chosen_dataset], train)
        , sep='\t', header=None)

    val_glob = pd.read_csv(
        os.path.join(dataset_base, datasets[chosen_dataset], val)
        , sep='\t', header=None)

    test_glob = pd.read_csv(
        os.path.join(dataset_base, datasets[chosen_dataset], test)
        , sep='\t', header=None)

    train_X = np.array(train_glob.loc[:, 0:1])
    train_X = np.array([[tokenizer.texts_to_sequences([txt])[0][0] for txt in row] for row in train_X]).squeeze()
    train_Y = np.array(train_glob.loc[:, 2])
    train_Y = train_Y * 1

    val_X = np.array(val_glob.loc[:, 0:1])
    val_X = np.array([[tokenizer.texts_to_sequences([txt])[0][0] for txt in row] for row in val_X]).squeeze()
    val_Y = np.array(val_glob.loc[:, 2])
    val_Y = val_Y * 1

    test_X = np.array(test_glob.loc[:, 0:1])
    test_X = np.array([[tokenizer.texts_to_sequences([txt])[0][0] for txt in row] for row in test_X]).squeeze()
    test_Y = np.array(test_glob.loc[:, 2])
    test_Y = test_Y * 1

    models = [model1, model2]
    testset_accs = list()
    testset_losses = list()
    for i, model in enumerate(models):
        history = EpochHistory()
        print("## Training model #%d ##" % i)
        model.fit(train_X, train_Y, epochs=100, validation_data=(val_X, val_Y), callbacks=[history], verbose=0)

        testset_accs.append(history.accs)
        testset_losses.append(history.losses)

        print("## Testing model #%d ##" % i)
        t_loss, t_acc = model.evaluate(test_X, test_Y, verbose=0)
        print("Test Results: loss=%f, acc=%f" % (t_loss, t_acc))

    accs.append(testset_accs)
    losses.append(testset_losses)

array_type = np.array(losses)
labels = ['Google w2v', 'Vanilla Embedding']
colors = ['r', 'g', 'y', 'c', 'm']
plt.xlabel(labels[0])
plt.ylabel(labels[1])
plt.plot([array_type.min(), array_type.max()], [array_type.min(), array_type.max()])

for i, show_array in enumerate(array_type):
    show_array = np.array(show_array).T

    plt.plot(show_array[:, 0], show_array[:, 1], colors[i]+'o')
    plt.show()