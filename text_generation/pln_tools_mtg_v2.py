import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import math
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import sklearn.decomposition as decomposition



def cardRegexFilter(carta): 
    
    carta['text'] = re.sub("{.*}: "," ",carta['text'])
    carta['text'] = re.sub("{.*}, "," ",carta['text'])
    carta['text'] = re.sub("{.*} "," ",carta['text'])
    carta['text'] = re.sub("{.*}"," ",carta['text'])
    carta['text'] = re.sub(".*[0-9]: "," ",carta['text'])
    carta['text'] = re.sub("â€”"," ",carta['text'])
    carta['text'] = re.sub("â€¢"," ",carta['text'])
    carta['text'] = re.sub("[A-z]*'[A-z]*"," ",carta['text'])
    carta['text'] = re.sub(" [0-9]* "," ",carta['text'])
    carta['text'] = re.sub("{[0-9]}"," ",carta['text'])
    carta['text'] = re.sub(" \+./\+. "," ",carta['text'])
    carta['text'] = re.sub(" \-./\-. "," ",carta['text'])
    carta['text'] = re.sub(" \+./\-. "," ",carta['text'])
    carta['text'] = re.sub(" \-./\+. "," ",carta['text'])
    carta['text'] = re.sub(" [0-9]+/[0-9]+ "," ",carta['text'])
    carta['text'] = re.sub(str(""+carta['name']+" ")," ", carta['text'])
    carta['text'] = re.sub(str(" "+carta['name']+" ")," ", carta['text'])
    carta['text'] = re.sub(str(""+carta['name']+".")," ", carta['text'])
    carta['text'] = re.sub(" . "," ",carta['text'])
    
    
    
def cardStopWordsFilter(carta, stopWords = set(stopwords.words("english"))):
    carta['text'] = ' '.join(w for w in (word_tokenize(carta['text'])) if not(w in list(stopWords)))

def cardPuntuacionFilter(carta, puncList):
    for punc in puncList:
        carta['text'] = carta['text'].replace(punc,'')

def cardRarityFilter(carta, rarities):
    carta['text'] = ' '.join(w for w in (word_tokenize(carta['text'])) if not(w in rarities))

def cardNameFilter(carta):
    carta['text'] = re.sub(str(""+carta['name']+" ")," ", carta['text'])
    carta['text'] = re.sub(str(" "+carta['name']+" ")," ", carta['text'])
    carta['text'] = re.sub(str(""+carta['name']+".")," ", carta['text'])
    
def cardMtgTextFilter(cartas, toRemoveWords, puncList, stopWords = set(stopwords.words("english"))):
    for carta in cartas:
        cardRegexFilter(carta)
        carta['text'] = carta['text'].lower()
        cardStopWordsFilter(carta, stopWords)
        cardRarityFilter(carta, rarities = ['``','{','}',"''"])
        cardPuntuacionFilter(carta, puncList)
        carta['text'] = ' '.join(word_tokenize(carta['text']))
        carta['text'] = removerPalabras(carta['text'], toRemoveWords)
        
def cardMtgTextFilter2(cartas, corpus):
    for carta in cartas:
        carta['text'] = removerPalabrasSinRepeticion(carta['text'], corpus)

def cardMtgCorpusFix(cartas):
    for carta in cartas:
        carta['text'] = ' '.join(word_tokenize(carta['text']))
        carta['text'] = carta['text'].lower()
   
def createCardCorpusText(cartas):
    corpus = ""
    for carta in cartas:
        corpus += str(carta['text']+" ")
    
    return corpus

def createCardCorpus(cartas):
    corpus = []
    for carta in cartas:
        corpus.append(' '.join(word_tokenize(carta['text'])))
    
    return corpus

def n_gramas(texto, n):
    return list(ngrams(word_tokenize(texto),n))

def removerPalabras(string, toRemoveWords):
    return ' '.join([word for word in word_tokenize(string, language = 'english') if not(word in toRemoveWords)])

def removerPalabrasSinRepeticion(string, corpus):
    corpus_tokenizado = word_tokenize(corpus, language = 'english') 
    return ' '.join([word for word in word_tokenize(string) if not(word in FreqDist(corpus_tokenizado).hapaxes())])


def tf_idf(termino, documento, corpus):
    aux = word_tokenize(documento['text'])
    tf = aux.count(termino)/len(aux)
    idf = math.log(len(corpus)/len([1 for carta in corpus if termino in word_tokenize(carta['text'])]))
    return tf*idf

def tokenizer(texto, n):
    return ' '.join([x[0]+"_"+x[1] for x in n_gramas(texto, n)])
    

def cleanDeckText(cartas, toRemoveWords, puncList):
    cardMtgTextFilter(cartas, toRemoveWords, puncList)
    corpus_aux = createCardCorpusText(cartas)
    cardMtgTextFilter2(cartas, corpus_aux)
    
def LDA(num_topics, num_top_words, deck):
    vectorizer = CountVectorizer(tokenizer=word_tokenize)
    X = vectorizer.fit_transform(deck)
    X_vocab = np.array(vectorizer.get_feature_names())
    lda = decomposition.LatentDirichletAllocation(n_topics=num_topics,learning_method='online')
    lda.fit_transform(X)
    lda_topic_words = []
    for topic in lda.components_:
        word_idx = np.argsort(topic)[::-1][0:num_top_words]
        lda_topic_words.append([X_vocab[i] for i in word_idx])
    
    lda_topic_words
        
    
def NMF(num_topics, num_top_words, deckCorpus, deck):
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
    X = vectorizer.fit_transform(deckCorpus)
    X_vocab = np.array(vectorizer.get_feature_names())
    nmf = decomposition.NMF(n_components=num_topics)
    nmf.fit_transform(X)
    nmf_topic_words = []
    for topic in nmf.components_:
        word_idx = np.argsort(topic)[::-1][0:num_top_words]
        nmf_topic_words.append([X_vocab[i] for i in word_idx])
    
    return nmf_topic_words



