from data import ScisummData, VenduData, AANData, GetBow, GetTaggedDocs
from pathlib import Path
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.spatial.distance import cdist
from analysis import ScisummAnalysis
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

class BertModel:
    """
    Base class for LSA model.
    """
    def __init__(self, base_model):
        """
        Initialize model with parameters. Model is fit if it has not been done before.

        :param base_model: Which Bert pretrained model to use.
        """

        self.shortname = 'S-BERT'
        self.name = base_model
        self.path = None
        self.model = None
        self.doc_vecs = None

    def fit(self, data):
        """
            Fit LSA model to the data, set document topic vectors and calculate distances.
        """

        self.path = Path('modelfiles/%s/%s' % (data.name, self.name))
        self.path.mkdir(parents=True, exist_ok=True)

        try:
            self.doc_vecs = pd.read_csv(self.path / str('document_vectors_%s.csv' % data.name), index_col=0)
        except:

            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.name)

            print("Fitting model...", end='')
            time.sleep(0.1)

            # Container for document topic vectors with zeros
            doc_vecs = np.zeros((len(data.ids), 768))

            start = time.time()
            # For each document
            for i in range(len(data.ids)):

                #if i % 100 == 0:
                minutes = (time.time() - start) // 60
                seconds = (time.time() - start) % 60
                print('\r%i / %i\t%i:%i' % (i, len(data.ids), minutes, seconds), end='')

                id = data.ids[i]
                document = data.get_sents_by_id(id)
                vectors = self.model.encode(document)

                # Set document vector
                doc_vecs[i] = np.mean(vectors, axis=0)

            # Set document topic vectors as pandas dataframe
            self.doc_vecs = pd.DataFrame(doc_vecs, index=data.ids)
            self.doc_vecs.to_csv(self.path / str('document_vectors_%s.csv' % data.name))

