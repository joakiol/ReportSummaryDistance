from data import ScisummData, VenduData, AANData, GetBow, GetTaggedDocs
from pathlib import Path
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.spatial.distance import cdist
from analysis import ScisummAnalysis
import pandas as pd
import numpy as np
import time
from scipy.spatial.distance import cosine
from tqdm import tqdm
import random

class TFIDFModel:
    """
    Base class for TFIDF model.
    """
    def __init__(self):
        """
        Initialize model with parameters. Model is fit if it has not been done before.

        """
        self.shortname = 'TF-IDF'
        self.name = 'TF-IDF'
        self.word_dict = None
        self.model = True
        self.path = None
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

    def fit(self, data):
        """
            Fit TFIDF model to the data, set document topic vectors and calculate distances.
        """

        self.path = Path('modelfiles/%s/%s' % (data.name, self.name))
        self.path.mkdir(parents=True, exist_ok=True)

        # If data is vendu, document vectors can't be made, since this takes an awful lot of space.
        # This therefore calculates distances directly
        if data.name == 'Vendu':
            self.doc_vecs = None

            try:
                distances = pd.read_csv(self.path / str('distances_%s_%s.csv' % (data.name, 'cosine')), index_col=0)
            except:
                print("Calculating distances...", end='')
                time.sleep(0.1)

                distances = np.zeros((len(data.stem_ids), 2))
                datastream = GetBow(data, self.remove_stopwords, self.word_dict)

                # For each document
                for i in range(len(datastream)):
                    name = data.ids[i]
                    if name.split('_')[-1] == 'abstract':

                        # Make vectors for report and abstract
                        abstract = np.zeros(len(self.word_dict))
                        for element in datastream[i]:
                            abstract[element[0]] = element[1]

                        report = np.zeros(len(self.word_dict))
                        for element in datastream[i+1]:
                            report[element[0]] = element[1]

                        # Search random until an abstract is found. Choose as other
                        other = np.zeros(len(self.word_dict))
                        while True:
                            randint = random.randint(0, len(data.ids)-1)
                            randname = data.ids[randint]
                            if randname.split('_')[-1] == 'abstract' and randint%100 != 0:
                                for element in datastream[randint]:
                                    other[element[0]] = element[1]
                                break

                        # Calculate distances
                        distances[int(i/2)][0] = cosine(report, abstract)
                        distances[int(i/2)][1] = cosine(report, other)

                # Save distances to CSV. We now have distances, without having to store 100k x 50k document vectors
                distances[np.isnan(distances)] = 1
                distances = pd.DataFrame(distances, index=data.stem_ids, columns=['own', 'other'])
                distances.to_csv(self.path / str('distances_%s_%s.csv' % (data.name, 'cosine')))

        # If data is not vendu, get document vectors as usual.
        else:

            try:
                self.doc_vecs = pd.read_csv(self.path / str('document_vectors_%s.csv' % data.name), index_col=0)
            except:

                print("Fitting model...", end='')
                time.sleep(0.1)

                # Container for document topic vectors with zeros
                doc_vecs = np.zeros((len(data.ids), len(self.word_dict)))

                datastream = GetBow(data, self.remove_stopwords, self.word_dict)


                # For each document
                for i in range(len(datastream)):

                    # Element is now tuple of (index, value)
                    for element in datastream[i]:
                        doc_vecs[i][element[0]] = element[1]

                # Set document topic vectors as pandas dataframe
                self.doc_vecs = pd.DataFrame(doc_vecs, index=data.ids)
                self.doc_vecs.to_csv(self.path / str('document_vectors_%s.csv' % data.name))

