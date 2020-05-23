import plaidml.keras
plaidml.keras.install_backend()

import pickle
import pln_tools_comentarios_vtg as pln
import numpy as np

from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint
from numpy import array
from pickle import dump
from nltk.tokenize import word_tokenize
from pln_tools_comentarios_vtg import removerSignos




if __name__ == '__main__':
	
	STEP = 1
	SEQUENCE_LEN = 30
	BATCH_SIZE = 128


	corpus = pickle.load(open("corpus_mtg_afterFilters_allCards.p", "rb"))

	text = [removerSignos(documento, ['{','}','.', ':', '(', ')', ',']) for documento in corpus]

	text = pln.createCorpusText(text) #String con el texto completo del corpus
	text_tokenizado = word_tokenize(text)


	# cut the text in semi-redundant sequences of SEQUENCE_LEN words
	sentences = []
	next_words = []
	for i in range(0, len(text_tokenizado) - SEQUENCE_LEN, STEP):
		sentences.append(text_tokenizado[i: i + SEQUENCE_LEN])
		next_words.append(text_tokenizado[i + SEQUENCE_LEN])


	sentences = [' '.join(sentence) for sentence in sentences]
	corpus_ = corpus[:]
	corpus_.append(' '.join(next_words))
	v = pln.Vectorizer()
	vocab = v.makeVocab(corpus)
	X = v.wordVectorizer(sentences,vocab,SEQUENCE_LEN)
	y = v.wordVectorizer(next_words,vocab,1)


	#y = np_utils.to_categorical(y)
	keylist = list(vocab.keys())

	
	# define model
	model = Sequential()
	model.add(Embedding(len(vocab), 50, input_length=SEQUENCE_LEN))
	model.add(LSTM(100, return_sequences=True))
	model.add(LSTM(100))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(len(vocab), activation='softmax'))
	print(model.summary())


	filepath="mtg_generator_weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
	#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	#callbacks_list = [checkpoint]
	# compile model
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
	#model.fit(X, y, batch_size=BATCH_SIZE, epochs=1, callbacks=callbacks_list)
	model.fit(X, y, batch_size=BATCH_SIZE, epochs=4)
	# save the model to file
	model.save('model3.h5')
	
	"""
	model = load_model('model2.h5')
	seed_text = np.reshape(X[5000],(1,-1))

	print("\n Texto Semilla:")

	print(" ".join([keylist[i] for i in seed_text[0]]))

	texto_output = []


	for i in range(200):

		yhat = model.predict_classes(seed_text, verbose=0)
		seed_text = np.append(seed_text,yhat[0])
		seed_text = pad_sequences([seed_text], maxlen=SEQUENCE_LEN, truncating='pre')
		texto_output.append(yhat[0])

	print("\n Texto Generado:")
	print(" ".join([keylist[i] for i in texto_output]))
	"""
"""
print(len(keylist))
model = load_model('model.h5')
seed_text = np.reshape(X[0],(1,-1))
print(seed_text)

texto_output = seed_text


for i in range(50):

	yhat = model.predict_classes(seed_text, verbose=0)
	print(yhat)
	print(len(yhat))
	print(keylist[yhat[0]])
	print(np.asarray(seed_text).shape)
	seed_text = np.append(seed_text,yhat[0])
	seed_text = pad_sequences([seed_text], maxlen=SEQUENCE_LEN, truncating='pre')
	print(seed_text)
	texto_output = np.append(texto_output,yhat)


print(" ".join([keylist[i] for i in texto_output]))
"""

"""



# define model
model = Sequential()
model.add(Embedding(len(vocab), 50, input_length=SEQUENCE_LEN))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(len(vocab), activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=BATCH_SIZE, epochs=1)
 
# save the model to file
model.save('model.h5')
"""