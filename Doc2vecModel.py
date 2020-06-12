from data import ScisummData, VenduData, AANData, GetBow, GetTaggedDocs
from pathlib import Path
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.spatial.distance import cdist
from analysis import ScisummAnalysis
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

class Doc2vecModel:
    """
    Base class for LSA model.
    """
    def __init__(self, vector_length, type='DM', window = 2, min_count = 1):
        """
        Initialize model with parameters. Model is fit if it has not been done before.

        :param vector_length: Number of topics in model.
        :param window: Window size for PV-DM
        :param min_count: Minimum number of times a word must appear to be considered.
        """

        if type == 'DM':
            self.dm = 1
        else:
            self.dm = 0

        self.shortname = 'Doc2vec'
        self.name = 'Doc2vecmodel%s_dm%s_w%s_mc%s' % (str(vector_length), str(self.dm), str(window), str(min_count))
        self.vector_length = vector_length
        self.window = window
        self.min_count = min_count
        self.path = None
        self.model = None
        self.doc_vecs = None
        self.infer = False


    def make_vocabulary(self, data):

        ids = []
        indicator = []
        for i in range(len(data)):
            ids.extend(data[i].ids)
            indicator.extend([i]*len(data[i].ids))

        def generator():
            print("")
            start = time.time()
            for i in range(len(ids)):
                if i % 100 == 0:
                    minutes = (time.time() - start) // 60
                    seconds = (time.time() - start) % 60
                    print('\r%i / %i\t%i:%i' % (i, len(ids), minutes, seconds), end='')
                yield TaggedDocument(data[indicator[i]].get_words_by_id(ids[i], True, False), [i])

        if self.model == None:
            self.model = Doc2Vec(vector_size=self.vector_length, window=self.window,
                                 min_count=self.min_count, workers=4, dm=self.dm)
        else:
            print("Vobabulary can't be built after training. This call does nothing")
            return

        print("Building vocabulary...", end='')
        time.sleep(0.1)

        self.model.build_vocab(generator())





    def train(self, data, epochs, infer = False):
        """
        Fit LSA model to the data, set document topic vectors and calculate distances.

        :param data: Data to fit model on
        """

        self.infer = infer

        if self.model != None:
            print("Model has already been trained. Doc2vec can only be trained once. Call does nothing. ")
            return
        for i in data:
            self.name += '_%s' % i.name
        self.name = '%s_e%s' % (self.name, str(epochs))
        self.path = Path('modelfiles/%s/%s' % (data[-1].name, self.name))

        try:
            self.model = Doc2Vec.load(str(self.path / 'model'))
        except:
            self.path.mkdir(parents=True, exist_ok=True)

            datastream = GetTaggedDocs(data)

            print("Training model...", end='')
            self.model = Doc2Vec(datastream, vector_size=self.vector_length, window=self.window,
                                 min_count=self.min_count, workers=4, epochs=epochs, dm=self.dm)

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

            if self.infer == True:
                j = 0
                for words in GetTaggedDocs([data]):
                    doc_vecs[j] = self.model.infer_vector(words[0])
                    j += 1

            else:
                # For each document
                for i in range(len(data.ids)):
                    # Set document vector
                    doc_vecs[i] = self.model.docvecs[data.ids[i]]



            # Set document topic vectors as pandas dataframe
            self.doc_vecs = pd.DataFrame(doc_vecs, index=data.ids)
            self.doc_vecs.to_csv(self.path / str('document_vectors_%s.csv' % data.name))

