from data import ScisummData, VenduData, AANData, GetBow, GetTaggedDocs, GetSents
from pathlib import Path
from gensim.models.word2vec import Word2Vec
from scipy.spatial.distance import cdist
from analysis import ScisummAnalysis
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

class Word2vecModel:
    """
    Base class for LSA model.
    """
    def __init__(self, vector_length, type='SG', window = 2, min_count = 1):
        """
        Initialize model with parameters. Model is fit if it has not been done before.

        :param vector_length: Number of topics in model.
        :param window: Window size for PV-DM
        :param min_count: Minimum number of times a word must appear to be considered.
        """

        if type == 'SG':
            self.sg = 1
        else:
            self.sg = 0


        self.shortname = 'Word2vec'
        self.name = 'Word2vecmodel%s_sg%s_w%s_mc%s' % (str(vector_length), str(self.sg), str(window), str(min_count))
        self.vector_length = vector_length
        self.window = window
        self.min_count = min_count
        self.path = None
        self.model = None
        self.doc_vecs = None

    def train(self, data, epochs):
        """
        Fit LSA model to the data, set document topic vectors and calculate distances.

        :param data: Data to fit model on
        """

        if self.model != None:
            print("Model has already been trained. Doc2vec can only be trained once. Call does nothing. ")
            return

        for i in data:
            self.name += '_%s' % i.name
        self.name = '%s_e%s' % (self.name, str(epochs))
        self.path = Path('modelfiles/%s/%s' % (data[-1].name, self.name))

        try:
            self.model = Word2Vec.load(str(self.path / 'model'))
        except:
            self.path.mkdir(parents=True, exist_ok=True)

            datastream = GetSents(data)

            print("Training model...", end='')
            self.model = Word2Vec(datastream, size=self.vector_length, window=self.window,
                                 min_count=self.min_count, workers=4, sg=self.sg, iter=epochs)

            self.model.save(str(self.path / 'model'))

    def fit(self, data):
        """
            Fit LSA model to the data, set document topic vectors and calculate distances.
        """

        if self.model == None:
            print("Model must be trained first. This function call does nothing")
            return

        try:
            self.doc_vecs = pd.read_csv(self.path / str('document_vectors_%s.csv' % data.name), index_col=0)
        except:

            print("Fitting model...", end='')
            time.sleep(0.1)

            # Container for document topic vectors with zeros
            doc_vecs = np.zeros((len(data.ids), self.vector_length))

            # For each document
            j = 0

            from nltk.corpus import stopwords
            if data.name == 'Vendu':
                stop_words = stopwords.words('norwegian')
            else:
                stop_words = stopwords.words('english')


            for doc in GetTaggedDocs([data]):
                vector = np.zeros(self.vector_length)
                words = 0
                for word in doc[0]:
                    if word not in stop_words:
                        try:
                            vector += self.model.wv[word]
                            words += 1
                        except:
                            continue

                doc_vecs[j] = vector/words
                j += 1



            # Set document topic vectors as pandas dataframe
            self.doc_vecs = pd.DataFrame(doc_vecs, index=data.ids)
            self.doc_vecs.to_csv(self.path / str('document_vectors_%s.csv' % data.name))

