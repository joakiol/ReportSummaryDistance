from data import ScisummData, VenduData, AANData, GetBow
from pathlib import Path
from gensim.models import LsiModel
from scipy.spatial.distance import cdist
from analysis import ScisummAnalysis
import pandas as pd
import numpy as np
import time

class LSAModel:
    """
    Base class for LSA model.
    """
    def __init__(self, vector_length):
        """
        Initialize model with parameters. Model is fit if it has not been done before.

        :param vector_length: Number of topics in model.
        """

        self.shortname='LSA'
        self.name = 'LSAmodel' + str(vector_length)
        self.vector_length = vector_length
        self.remove_stopwords = None
        self.word_dict = None
        self.path = None
        self.model = None
        self.doc_vecs = None

    def set_dict(self, data, remove_stopwords = False, no_below = 1, no_above = 1, filter_most_frequent = 0):
        """
        Set/make dictionary to be used for bow representations.

        :param data: Which data to use for making dictionary.
        :param remove_stopwords: Whether to remove stopwords.
        :param no_below: Minimum number of documents a word has to appear in to be included.
        :param no_above: Maximum fraction of documents a word can appear in to be included.
        :param filter_most_frequent: Remove the most frequent words.
        """

        if self.word_dict != None:
            print("Model already have a dictionary! This function call does nothing. ")
            return

        self.name = '%s_%sdict_rs%s_nb%s_na%s_fmf%s' % (self.name, data.name, str(remove_stopwords), str(no_below),
                                                        str(no_above), str(filter_most_frequent))

        self.remove_stopwords = remove_stopwords
        self.word_dict = data.get_dictionary(remove_stopwords, no_below, no_above, filter_most_frequent)

    def train(self, data):
        """
        Fit LSA model to the data, set document topic vectors and calculate distances.

        :param data: Data to fit model on
        """

        if self.word_dict == None:
            print("Dictionary must be assigned to model before training. This function call does nothing")
            return
        if self.model == None:
            self.model = LsiModel(num_topics=self.vector_length, id2word=self.word_dict)

        self.name = '%s_%strain' % (self.name, data.name)
        self.path = Path('modelfiles/%s/%s' % (data.name, self.name))

        try:
            self.model = LsiModel.load(str(self.path / '.model'))
        except:
            self.path.mkdir(parents=True, exist_ok=True)

            print("Training model...", end='')
            time.sleep(0.1)

            datastream = GetBow(data, self.remove_stopwords, self.word_dict)
            self.model.add_documents(datastream)

            self.model.save(str(self.path / '.model'))

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
            datastream = GetBow(data, self.remove_stopwords, self.word_dict)
            for i in range(len(datastream)):

                # element is now a tuple with index and value for nonzero vector elements
                for element in self.model[datastream[i]]:

                    # Set nonzero elements in container
                    doc_vecs[i][element[0]] = element[1]

            # Set document topic vectors as pandas dataframe
            self.doc_vecs = pd.DataFrame(doc_vecs, index=data.ids)
            self.doc_vecs.to_csv(self.path / str('document_vectors_%s.csv' % data.name))

