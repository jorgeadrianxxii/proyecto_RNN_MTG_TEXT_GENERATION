'''
Created on 25 abr. 2018

@author: Adrian
'''

import re
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.stats.mstats import gmean
from math import isnan
import numpy as np
import sklearn.decomposition as decomposition
import string
import numbers


def createCorpusText(documentos):
    corpus = ""
    for texto in documentos:
        corpus += str(texto+" ")
    
    return corpus


def promedio_tokens(documentos):
    tokens_por_documento = [len(word_tokenize(documento)) for documento in documentos] 
    return np.mean(tokens_por_documento), gmean(tokens_por_documento)

def removerPalabrasSinRepeticion(documentos): 
    palabrasNoRep = FreqDist(word_tokenize(createCorpusText(documentos))).hapaxes()
    return [' '.join(word for word in word_tokenize(texto) if not (word in palabrasNoRep)) for texto in documentos]

def removerPalabras(documentos, toRemoveWords):
    return [' '.join(word for word in word_tokenize(texto) if not (word in toRemoveWords)) for texto in documentos] 

def removerAcentos(documento):
    palabras = documento.split()
    acentos = { "á":"a",  "é":"e", "í": "i", "ó":"o", "ú":"u"}
    dic = list(zip(acentos.keys(), acentos.values()))
    llaves = acentos.keys()
    for palabra, auxCont in zip(palabras,range(len(palabras))):
        letras = list(palabra)
        for letra, cont2 in zip(letras,range(len(letras))):
            if(letra in llaves):
                for tupla in dic:
                    if(letra == tupla[0]):
                        letras[cont2] = tupla[1]
                        palabras[auxCont] = ''.join(letras)
    
    return " ".join(palabras)

def removerEmojis(documento):
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        
    return emoji_pattern.sub( r'', documento ) 

def removerEmail(documento):
    text = re.sub(r'[\w\.-]+@[\w\.-]+ ', "", documento )
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))* ', '',  text )
    return text

def removerUserName(documento):
    return re.sub( r'@[a-zA-Z0-9-_.]+[a-zA-Z0-9]*', "", documento)

def removerSignos(documento, puncList):
    text = documento
    #puncList = ['@','|','"','|...', '``', '-', '”','“',"¡","¿", ".", ";", ":", "!", "?", "/", "\\", ",", ")", "(", "\"", ']', 
    #                '[', "''", '…', '5', 'ì', '️', 'â', 'ò', '#', '$', 'ė', '´']
    
    for pun in puncList:
        if pun in text:
            text = text.replace( pun, '' )
    return text

def minCharacters(documento, min=2):
    text = [palabra for palabra in documento.split() if(len(palabra)>=min)]
    """for palabra in palabras:
        if(len(palabra)>1):
            text.append(palabra)
    """        
    text = ' '.join(text)
    return text


def limpiar_documentos(documentos):
    documentos_limpios = []
    for documento in documentos:
        texto = removerSignos(documento)
        texto = removerAcentos(texto)
        texto = minCharacters(texto)
        documentos_limpios.append(texto)

    toRemoveWords = ['que', 'para', 'pero', 'tus', 'dtb', 'los', 'las', 'una', 'como', 'muy', 'todos','todas', 'del', 'tan', 
                     'sea', 'ser', 'con', 'por', 'les', 'uno', 'dos', 'donde', '%', '100', '6672', 'ala',
                     'bla', 'bcs', 'esa', 'la', 'es', 'de', 'el', 'hay','le','se']
    
    documentos_limpios = removerPalabrasSinRepeticion(documentos_limpios)
    documentos_limpios = removerPalabras(documentos_limpios, toRemoveWords)

    return documentos_limpios

def limpiar_comentarios(comentarios):
    comentarios_limpios = []
    
    for comentario in comentarios:
    
        palabras = comentario.split()
        
        #----------eliminar acentos
        acentos = { "á":"a",  "é":"e", "í": "i", "ó":"o", "ú":"u"}
        dic = list(zip(acentos.keys(), acentos.values()))
        llaves = acentos.keys()
        for palabra, auxCont in zip(palabras,range(len(palabras))):
            letras = list(palabra)
            for letra, cont2 in zip(letras,range(len(letras))):
                if(letra in llaves):
                    for tupla in dic:
                        if(letra == tupla[0]):
                            letras[cont2] = tupla[1]
                            palabras[auxCont] = ''.join(letras)
        
        text = " ".join( palabras )
        
        # eliminar emojis
        emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        
        text = emoji_pattern.sub( r'', text ) 
        
        #----------eliminar mayusculas
        text = text.lower()
        
        #----------remplazo username
        text = re.sub( '@lopezobrador_', "lopez obrador", text )
        
        #----------eliminamos email
        text = re.sub(r'[\w\.-]+@[\w\.-]+ ', "", text )
        text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))* ', '',  text )
        
        #----------eliminamos usuarios con arrobas @
        text = re.sub( r'@[a-zA-Z0-9-_.]+[a-zA-Z0-9]*', "", text)
        
        #----------eliminar hashtag #
        text = re.sub( r'#[a-zñA-Z0-9-_.]+[a-zA-Z0-9]*', "", text )
        
        #----------eliminar links
        text = re.sub( r'https://[a-zA-Z0-9-_./]+[a-zA-Z0-9]*', "", text )
        
        #----------eliminar RT
        text = re.sub("rt ", " ",text)
        
        
        
        
        #----------eliminar los signos de puntuacion
        puncList = ['@','|','"','|...', '``', '-', '”','“',"¡","¿", ".", ";", ":", "!", "?", "/", "\\", ",", ")", "(", "\"", ']', 
                    '[', "''", '…', '5', 'ì', '️', 'â', 'ò', '#', '$', 'ė', '´']
        for pun in puncList:
            if pun in text:
                text = text.replace( pun, '' )
                
        #----------minima longitud de una palabra
        palabras = text.split()
        text = []
        for palabra in palabras:
            if(len(palabra)>1):
                text.append(palabra)
                
        text = ' '.join(text)
        array_text = text.split()
        comentarios_limpios.append(' '.join(array_text))
    
    toRemoveWords = ['que', 'para', 'pero', 'tus', 'dtb', 'los', 'las', 'una', 'como', 'muy', 'todos', 'del', 'tan', 
                     'asi', 'sea', 'ser', 'con', 'por', 'les', 'pues', 'uno', 'dos', 'donde', '%', '100', '6672', 'ahi', 'ala',
                     'bla', 'bcs', 'esa']
    
    comentarios_limpios = removerPalabrasSinRepeticion(comentarios_limpios)
    comentarios_limpios = removerPalabras(comentarios_limpios, toRemoveWords)
    return comentarios_limpios


def NMF(num_topics, num_top_words, documentos):
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
    X = vectorizer.fit_transform(documentos)
    for row in X.todense().tolist():
        print(row)
    X_vocab = np.array(vectorizer.get_feature_names())
    print(X_vocab)
    nmf = decomposition.NMF(n_components=num_topics)
    nmf.fit(X)
    nmf_topic_words = []
    for topic in nmf.components_:
        word_idx = np.argsort(topic)[::-1][0:num_top_words]
        nmf_topic_words.append([X_vocab[i] for i in word_idx])
    
    return nmf_topic_words

def TfidfMatrix(documentos):
    matrix = []
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
    X = vectorizer.fit_transform(documentos)
    for row in X.todense().tolist():
        matrix.append(row)
    X_vocab = np.array(vectorizer.get_feature_names())
    return matrix, X_vocab


class TfidfMatrixSnack:
    def __init__(self, tfidf_matrix,numEntradas,etiquetas, sort=False):
        fullymatrixSnack = []
        for index, row in enumerate(tfidf_matrix):
            cont = 0
            pesostfidf = [0 for _ in range(numEntradas)]
            if(sort == True):
                row.sort(reverse = True)
            for x in row:
                if(cont>=numEntradas):
                    break
                pesostfidf[cont] = x
                cont = cont+1
            finalRow = pesostfidf
            finalRow.append(etiquetas[index])
            fullymatrixSnack.append(tuple(finalRow))
       
        self.matrixSnack = fullymatrixSnack



class InfoMatrixForRow:
    

    def __init__(self, TfidfMatrix, imprimir = False):
        self.info_for_row = []
        self._min_ = 0
        self._max_ = 0
        self._maxLenTuples_ = 0
        for row in TfidfMatrix:
            tuples = [(i,e) for i, e in enumerate(row) if e != 0.0]
            nonZeroIndexes,nonZeroValues = [0],[0]
            maxLenTuples = len(tuples)
            if(maxLenTuples>0):
                nonZeroIndexes,nonZeroValues = (zip(*tuples))
            minAux, maxAux, =  min(nonZeroValues),max(nonZeroValues) 
            if(minAux < self._min_):
                self._min_ = minAux
            if(maxAux > self._max_):
                self._max_ = maxAux
            if(maxLenTuples > self._maxLenTuples_):
                self._maxLenTuples_ = maxLenTuples
            self.info_for_row.append((nonZeroIndexes, nonZeroValues, maxLenTuples))
        
    def imprimir_info(self):
        for row, info in enumerate(self.info_for_row):
            print("Fila "+str(row)+", Número de Elementos diferentes de cero: "+str(info[2])+" Valor Máximo: "+str(max(info[1])) + " Valor Mínimo: "+ str(min(info[1])))
        
                

def word2Float(word):
    abc = list(enumerate(list(string.ascii_lowercase)))
    word_ = list(enumerate(list(word)))
    suma = 0
    for letra in word_:
        for i in abc:
            if(i[1] == letra[1]):
                suma = suma + (1/((i[0]*50)+50))*(letra[0]+1)
                break
    return suma
    

def emptyValueFilter(listTuples):
    newListTuples = []
    for tupla in listTuples:
        flag = True
        for value in list(tupla):
            if(isinstance(value, str)):
                if(value == 'nan' or value == '' or value == ' ' or len(value) == 0):
                    flag = False
                    break
            else:
                if(isinstance(value, numbers.Number)):
                    if(isnan(value)):       
                        flag = False
                        break
                else:
                    flag = False
                    break
        if(flag):
            newListTuples.append(tupla)
    return newListTuples 
    

def characterVectorizer(documentos):
    
    vectorized_documents = []
    values = dict()
    for index, letter in enumerate(string.ascii_lowercase+'áéíóúñ'):
        values[letter] = index+1
    

    for documento in documentos:
        vector = []
        letras = [x for x in documento if x != ' ']
        vector = [values.get(letra) if letra in values.keys() else 0 for letra in letras]
        vectorized_documents.append(vector)
    
        
    max_length = len(sorted(vectorized_documents,key=len, reverse=True)[0])
    vectorized_documents = np.array([row+[0]*(max_length-len(row)) for row in vectorized_documents])

    return vectorized_documents

class Vectorizer:

    def __init__(self):
        pass
    """Construye un diccionario en el cual las llaves son las palabras del
    vocabulario y los valores son tuplas de la forma: (id, #instancias, #documentos)"""
    def makeVocab(self, documents):
        vocab = dict()
        lastID = 0
        for document in documents:
            words_cache = []
            for word in document.split():
                if(not(word in vocab.keys())):
                    vocab[word] = (lastID,1,1)
                    lastID = lastID + 1
                elif(not(word in words_cache)):
                    vocab[word] = (vocab[word][0],vocab[word][1]+1,vocab[word][2]+1)
                    words_cache.append(word)
                else:
                    vocab[word] = (vocab[word][0],vocab[word][1]+1,vocab[word][2])
        return vocab       
    #def tfidfMatrix(self, documents):
    #   if(len(self.vocab.keys) == 0):
    #        self.makeVocab(documents)

    def wordVectorizer(self, documents, vocab, max_length):
        vectorized_documents = []
        for document in documents:
            vector = [vocab.get(word)[0] if word in vocab.keys() else -1 for word in document.split()]
            vectorized_documents.append(vector)
            
        vectorized_documents = np.array([row+[0]*(max_length-len(row)) for row in vectorized_documents])

        return vectorized_documents