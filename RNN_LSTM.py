import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import h5py
import pickle
import os

CLUSTER_PATH = '/users/vsoni3/machinelearning/'
PATH = 'Applied ML/Project2/'
DATAFILE_NAME = 'updated_data_binary.csv'
IS_EMBEDDING = True
IS_GLOVE = False
IS_GLOVE_AVAIL = True

#setting model paraemters
epochs = 10
emb_dim = 100
batch_size = 32

#get data for news articles
#data = pd.read_csv('users/vsoni3/machinelearning/'+DATAFILE_NAME)
#data.head()

#import preprocessed text data
with open(CLUSTER_PATH+'posts.pickle','rb') as f:
    posts = pickle.load(f)

with open(CLUSTER_PATH+'output.pickle','rb') as f:
    output = pickle.load(f)
################################################################
# Preprocessing text
################################################################
max_len = 8000
embedding_dim = 100
#filter text and tokenizing
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(posts)
sequences = tokenizer_obj.texts_to_sequences(posts)

#pad sequences
word_index = tokenizer_obj.word_index
print('Found %s unique tokens. '%len(word_index))

print('Shape of output tensor:',np.array(output).shape)
vocab_size = len(word_size) + 1

###############################################################
# Glove word embedding 100 dimensions
# https://nlp.stanford.edu/projects/glove/
###############################################################
GLOVE_DIR = '/glove_6b/glove.6B/'
EMBED_MATRIX_DIR = ''

#checks if it has loaded embedding then import it
if IS_GLOVE_AVAIL:
    with open(CLUSTER_PATH+'embedding_matrix.pickle','rb') as s:
        embedding_matrix_glove = pickle.load(s)
else:
    #dictionary to store word embeddings
    embeddings_index = {}

    with open(os.path.join(PATH+GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32') # Load embedding
            embeddings_index[word] = embedding # Add embedding to our embedding dictionary

    print('Found {:,} word vectors in GloVe.'.format(len(embeddings_index)))

    word_index = tokenizer.word_index
    nb_words = min(vocab_size, len(word_index)) # How many words are there actually

    embedding_matrix_glove = np.zeros((nb_words, embedding_dim))

    # Loop over all words in the word index
    for word, i in word_index.items():
        # If we are above the amount of words we want to use we do nothing
        if i >= vocab_size:
            continue
        # Get the embedding vector for the word
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix_glove[i] = embedding_vector

    with open('embedding_matrix.pickle','wb') as s:
        pickle.dump(embedding_matrix, s)
################################################################
# Trained word2vec on data
################################################################
with open(CLUSTER_PATH+'w2vmatrix.pickle','rb') as f:
    embedding_matrix = pickle.load(f)

#One-hot encode the labels
data.loc[data['subreddit'] == 'conservative', 'LABEL'] = 0
data.loc[data['subreddit'] == 'liberal', 'LABEL'] = 1
labels = to_categorical(data['LABEL'], num_classes=2)

################################################################
# Network architecture and training
################################################################

learningrate = [0.1, 0.01]
inputlength = [1000,1200]
lr_dict = {}

'''
RNN-LSTM network which takes learningrate and inputlength as parameters to learn weights.
Embedding layer takes either pre-trained embedding or learns the weight vectors during the training.
NOTE - Do not call this function for larger sequences on local machine unless machine have huge memory.
'''
def lstm_network(lr, length):
    text_pad = pad_sequences(sequences, maxlen=length)
    print('Shape of post tensor:',text_pad.shape)
    model_name = str(lr)+str(length)+'_model'
    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(text_pad , np.array(output), test_size=0.2, random_state=42)

    #defining LSTM model
    model = Sequential()
    #if embedding is being used then add weight matrix to the embedding layer
    if IS_EMBEDDING:
        model.add(Embedding(vocab_size, emb_dim, embeddings_initializer=Constant(embedding_matrix), input_length=max_len, trainable=False))
    else:
        model.add(Embedding(vocab_size, emb_dim, input_length=length))
    model.add(SpatialDropout1D(0.5))
    model.add(LSTM(32, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(2, activation='softmax'))

    rms = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
    model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.25)

    #store history to plot loss and accuracy
    with open(CLUSTER_PATH+model_name+'_hist.pickle') as f:
        pickle.dump(history.history, f)
    #save model
    model.save_weights(CLUSTER_PATH+model_name+'.h5')

for lr in learningrate:
    for length in inputlength:
        lstm_network(lr, length)

accr = model.evaluate(X_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
