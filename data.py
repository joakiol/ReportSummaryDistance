from nltk.tokenize import word_tokenize, sent_tokenize
from pathlib import Path
import xml.etree.ElementTree as ET
from gensim.corpora.dictionary import Dictionary
from gensim.models import TfidfModel
from nltk.corpus import stopwords
from tqdm import tqdm
import time
from gensim.models.doc2vec import TaggedDocument
import pickle


class GetSents:
    """
    Memory friendly dataloader for sentences in a corpus. Implements __iter__ and __next__,
    and is therefore iterable, but not indexable.
    """
    def __init__(self, data):
        """
        Set parameters necessary for dataloader.

        :param data: Which data objects to read.
        """

        # Data is a list, models can train on more than 1 data
        self.data = data

        # self.ids will contain document ids for each document in all input data, while indicator will indicate
        # which data each id comes from. Remember that data is list of data objects.
        self.ids = []
        self.indicator = []
        for i in range(len(data)):
            self.ids.extend(data[i].ids)
            self.indicator.extend([i] * len(data[i].ids))

        # Various necessary parameters
        self.length = len(self.ids)
        self.time = None
        self.passno = 0
        self.id = None
        self.index = None
        self.docindex = None
        self.doc = None
        self.doclength = None


    def __iter__(self):
        """
        This is called when something stars looping through this data

        :return: Returns itself in a state where it is on the first element of the iteration
        """

        # Prints to keep track of which pass it is
        self.passno += 1
        self.time = time.time()
        print("\nPass number %i" % self.passno)

        # Get current document, tokenize into list of sentences
        self.index = 0
        self.id = self.ids[0]
        self.doc = self.data[self.indicator[self.index]].get_tokenized_sents_by_id(self.id, lower=True)
        self.docindex = 0
        self.doclength = len(self.doc)
        return self

    def __next__(self):
        """
        Gets the next element in iteration

        :return: Element in iteration.
        """

        # Progress "bar"
        if self.index % 100 == 0:
            minutes = (time.time()-self.time) // 60
            seconds = (time.time()-self.time) % 60
            print('\r%i / %i\t%i:%i' % (self.index, self.length, minutes, seconds), end='')

        # Raise exception when done looping
        if self.index == self.length:
            print("\n")
            raise StopIteration

        # This means we have reached the end of the document. A new document must be read from memory, and tokenized
        # into list of sentences.
        if self.docindex == self.doclength:
            self.id = self.ids[self.index]
            self.doc = self.data[self.indicator[self.index]].get_tokenized_sents_by_id(self.id, lower=True)
            self.index += 1
            self.docindex = 0
            self.doclength = len(self.doc)
            if self.doclength == 0:
                return next(self)

        # Get sentence at current position in document, increment sentence position
        sent = self.doc[self.docindex]
        self.docindex += 1

        return sent

class GetTaggedDocs:
    """
    Memory friendly dataloader for TaggedDocs objects, for doc2vec.
    """
    def __init__(self, data):
        """
        Set parameters necessary for dataloader.

        :param data: Which data objects to read.
        """

        # data is list of (possibly) several data objects
        self.data = data

        # self.ids will contain document ids for each document in all input data, while indicator will indicate
        # which data each id comes from. Remember that data is list of data object
        self.ids = []
        self.indicator = []
        for i in range(len(data)):
            self.ids.extend(data[i].ids)
            self.indicator.extend([i] * len(data[i].ids))

        # Initialize necessary parameters
        self.length = len(self.ids)
        self.time = None
        self.passno = 0
        self.id = None
        self.index = None

    def __len__(self):
        """
        :return: number of documents in dataloader.
        """
        return len(self.ids)

    def __iter__(self):
        """
        This is called when something starts looping through this data

        :return: Returns itself in a state where it is on the first element of the iteration
        """

        # Print pass number (this call implies new pass over data).
        self.passno += 1
        self.time = time.time()
        print("\nPass number %i" % self.passno)

        self.index = 0
        self.id = self.ids[0]
        return self

    def __next__(self):
        """
        Gets the next element in iteration
        :return: Element in iteration.
        """

        # Print progress every 100
        if self.index % 100 == 0:
            minutes = (time.time()-self.time) // 60
            seconds = (time.time()-self.time) % 60
            print('\r%i / %i\t%i:%i' % (self.index, self.length, minutes, seconds), end='')

        # Stop iteration when at the end of list
        if self.index == self.length:
            print("\n")
            raise StopIteration

        # Get document, increment its own state to next
        self.id = self.ids[self.index]
        words = self.data[self.indicator[self.index]].get_words_by_id(self.id, lower=True, remove_stopwords=False)
        self.index += 1

        # Return TaggedDocument
        return TaggedDocument(words=words, tags=[self.id])



class GetBow:
    """
    Memory friendly dataloader for Bag-of-words (TF-IDF)
    """
    def __init__(self, data, remove_stopwords, word_dict, type='bow'):
        """
        Initialize dataloader.
        :param data: Single data object
        :param remove_stopwords: True/false indicating whether stopwords should be removed
        :param word_dict: Which word dictionary to use.
        :param type: Not really used
        """
        self.data = data
        self.remove_stopwords = remove_stopwords
        self.length = len(data.ids)
        self.word_dict = word_dict
        self.tf_idf_model = TfidfModel(dictionary=word_dict, smartirs='nfc')
        self.time = None
        self.passno = 0
        self.type = type
        self.id = None

    def __len__(self):
        """
        :return: Number of documents in data.
        """
        return self.length

    def __getitem__(self, item):
        """
        Implements __getitem__, which makes the dataloader indexable (and iterable).

        :param item: Index of document to get.
        :return: TFIDF-representation of document at ids[index}.
        """

        # Out of bounds.
        if item < 0 or item >= self.length:
            print("\n")
            raise IndexError

        # Print new pass if the 0-th element is indexed.
        if item == 0:
            self.time = time.time()
            self.passno += 1
            print("\nPass number %i" % self.passno)

        # Print progress every 100.
        if item % 100 == 0:
            minutes = (time.time()-self.time) // 60
            seconds = (time.time()-self.time) % 60
            print('\r%i / %i\t%i:%i' % (item, self.length, minutes, seconds), end='')

        # Get words, make bow and return TF-IDF for current document.
        words = self.data.get_words_by_id(self.data.ids[item], lower=True, remove_stopwords=self.remove_stopwords)
        bow = self.word_dict.doc2bow(words)
        return self.tf_idf_model[bow]

class TxtCorpusReader:
    """
    Class for data set with format as one text file per document,
    with one sentence per line.
    """
    def __init__(self, name, path, ids, stop_words):
        """
        Initialize parameters.

        :param data: Data object for data that TxtCorpusReader will read. Contains fileids and paths.
        """

        self.name = name
        self.path = path
        self.ids = ids
        self.stop_words = stop_words

    def get_raw_by_id(self, id):
        """
        Get raw text of document by document id.

        :param id: document id (string)
        :return: Raw text of document with the given id.
        """

        path = self.path / str(id + '.txt')

        with path.open('r', encoding='utf-8') as f:
            document = f.read()

        return document

    def get_words_by_id(self, id, lower = False, remove_stopwords = False):
        """
        Get the words of a specific document, by document id, using
        nltk word_tokenize function.

        :param id: document id (string).
        :param lower: True if capital letters should be lowered.
        :param remove_stopwords: True if stopwords should be removed. 
        :return: Words of document with given id (list of strings).
        """

        path = Path('data/%s_processed' % self.name)
        name = '%s_words_%s_%s' % (id, str(lower), str(remove_stopwords))

        try:
            with open(path / name, 'rb') as fp:
                words = pickle.load(fp)
        except:

            # Get raw document
            raw = self.get_raw_by_id(id)

            # Remove lower and remove stopwords if remove_stopwords == True.
            if remove_stopwords:
                words = [word.lower() for word in word_tokenize(raw) if word.lower() not in self.stop_words]

            # Lower words if only lower == TRUE.
            elif lower:
                words = [word.lower() for word in word_tokenize(raw)]

            # Else, return as is.
            else:
                words = word_tokenize(raw)

            path.mkdir(parents=True, exist_ok=True)

            with open(path / name, 'wb') as fp:
                pickle.dump(words, fp)

        # Return tokenized list of strings(words) using nltk tokenize function
        return words

    def get_words(self, lower = False, remove_stopwords = False):
        """
        Get the words of all documents.

        :param lower: True if capital letters should be lowered.
        :param remove_stopwords: True if stopwords should be removed.
        :return: List of documents, where each document is a list of words.
        """

        for i in tqdm(range(len(self.ids))):
            yield self.get_words_by_id(self.ids[i], lower=lower, remove_stopwords=remove_stopwords)


    def get_sents_by_id(self, id):
        """
        Get the sentences of a specific document, by document id, as strings.

        :param id: document id (string).
        :return: List of sentences. A sentence is a string.
        """

        raw = self.get_raw_by_id(id)
        sentences = raw.splitlines()

        return sentences

    def get_long_strings_by_id(self, id, max=510):
        """
        Not really used. Maximizes input lengths to max, for BERT, in an attempt to feed maximum-length
        sentences to BERT.

        :param id: Document id
        :param max: Length of sequences to shape document into
        :return: list of long-sequences for document.
        """

        tokenized_sentences = self.get_tokenized_sents_by_id(id)

        done = False
        idx = 0
        container = []

        while not done:
            sub_container = []
            while len(sub_container)+len(tokenized_sentences[idx]) <= max:
                sub_container.extend(tokenized_sentences[idx])
                idx += 1


                if idx == len(tokenized_sentences):
                    done = True
                    break

            string = ""
            for word in sub_container:
                string += " " + word
            container.append(string)

        return container



    def get_tokenized_sents_by_id(self, id, lower = False, remove_stopwords = False):
        """
        Get the sentences of a specific document, by document id, as list of words
        using nltk word_tokenize function.

        :param id: id: document id (string).
        :param lower: True if capital letters should be lowered.
        :param remove_stopwords: True if stopwords should be removed.
        :return: List of sentences. A sentence is a list of words (strings).
        """

        path = Path('data/%s_processed' % self.name)
        name = '%s_sents_%s_%s' % (id, str(lower), str(remove_stopwords))

        try:
            with open(path / name, 'rb') as fp:
                tokenized_sents = pickle.load(fp)
        except:

            tokenized_sents = []

            for sentence in self.get_sents_by_id(id):

                # Remove lower and remove stopwords if remove_stopwords == True.
                if remove_stopwords:

                    # Set language
                    if self.name == 'Vendu':
                        stop_words = stopwords.words('norwegian')
                    else:
                        stop_words = stopwords.words('english')

                    tokenized_sents.append([word.lower() for word in word_tokenize(sentence)
                                            if word.lower() not in stop_words])

                # Lower words if only lower == TRUE.
                elif lower:
                    tokenized_sents.append([word.lower() for word in word_tokenize(sentence)])

                # Else, return as is.
                else:
                    tokenized_sents.append(word_tokenize(sentence))

            with open(path / name, 'wb') as fp:
                pickle.dump(tokenized_sents, fp)

        return tokenized_sents

    def get_tokenized_sents(self, lower = False, remove_stopwords = False):
        """
        Get the sentences of all documents.

        :param lower: True if capital letters should be lowered.
        :param remove_stopwords: True if stopwords should be removed.
        :return: Sentences  of entire corpus
        """

        sentences = []

        for id in self.ids:
            sentences.extend(self.get_tokenized_sents_by_id(id, lower=lower, remove_stopwords=remove_stopwords))

        return sentences

    def get_dictionary(self, remove_stopwords, no_below, no_above, filter_most_frequent):
        """
        Get dictionary for corpus
        :param remove_stopwords: True if stopwords should be removed.
        :param no_below: Minimum number of documents a word has to appear in to be included.
        :param no_above: Maximum fraction of documents a word can appear in to be included.
        :param filter_most_frequent: Remove the most frequent words.
        :return: Dictionary
        """
        default_name = 'rm' + str(remove_stopwords)  + '_nb1_na1_fmf0.dict'

        dict_name = 'rm' + str(remove_stopwords)  + '_nb' + str(no_below) + \
                    '_na' + str(no_above) + '_fmf' + str(filter_most_frequent) + '.dict'
        dict_path = Path(str('modelfiles/' + self.name + '/dictionaries'))

        try:
            word_dict = Dictionary.load(str(dict_path / dict_name))

        except:

            try:
                word_dict = Dictionary.load(str(dict_path / default_name))
            except:
                dict_path.mkdir(parents=True, exist_ok=True)
                print("Making Dictionary...")
                time.sleep(0.1)
                word_dict = Dictionary(self.get_words(lower=True, remove_stopwords=remove_stopwords))
                word_dict.save(str(dict_path / default_name))

            # Make a Dictionary of words (map between position in b-o-w and word)
            word_dict.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
            word_dict.filter_n_most_frequent(filter_most_frequent)

            word_dict.save(str(dict_path / dict_name))

        return word_dict

    def get_bow(self, word_dict, remove_stopwords = False, no_below = 1, no_above = 1, filter_most_frequent = 0):
        """
        Get bag of word dictionary and bag of words representation of data.

        :param remove_stopwords: True if stopwords should be removed.
        :param no_below: Minimum number of documents a word has to appear in to be included.
        :param no_above: Maximum fraction of documents a word can appear in to be included.
        :param filter_most_frequent: Remove the most frequent words.
        :param type_bow: If other than TF-IDF should be used, this must be implemented
        :return: Bow representation of corpus, and word dictionary
        """

        #word_dict = self.get_dictionary(remove_stopwords, no_below, no_above, filter_most_frequent)

        tf_idf_model = TfidfModel(dictionary=word_dict, smartirs='nfc')

        for doc in self.get_words(lower=True, remove_stopwords=remove_stopwords):
            bow = word_dict.doc2bow(doc)
            yield tf_idf_model[bow]



class VenduData(TxtCorpusReader):
    """
    Class for Vendu data. Inherits from TxtCorpusReader,
    with a few corpus specific variables and methods.
    """

    def __init__(self):
        """
        Initialize parameters
        """

        # Containers for document ids
        ids = []
        self.report_ids = []
        self.abstract_ids = []
        self.stem_ids = []

        # Search data folder for document ids, add to containers.
        for fn in Path("data/Vendu_singles2").iterdir():
            name = fn.stem
            nametype = name.split('_')
            if nametype[2] == 'report':
                self.report_ids.append(name)
                self.stem_ids.append('%s_%s' % (nametype[0], nametype[1]))
            else:
                self.abstract_ids.append(name)
            ids.append(name)

        # Set super parameters
        TxtCorpusReader.__init__(self, name='Vendu', path=Path('data/Vendu_singles2'), ids=ids,
                                 stop_words = stopwords.words('norwegian'))


class ScisummData(TxtCorpusReader):
    """
    Class for ScisummNet-2019 data. Inherits from TxtCorpusReader,
    with a few corpus specific variables and methods.
    """

    def __init__(self):
        """
        Initialize parameters
        """

        # Containers for document ids
        ids = []
        self.report_ids = []
        self.abstract_ids = []

        # Search data folder for document ids, add to containers.
        for fn in Path("data/ScisummNet-2019_singles").iterdir():
            name = fn.stem
            nametype = name.split('_')[1]
            if nametype == 'report':
                self.report_ids.append(name.split('.')[0])
            else:
                self.abstract_ids.append(name.split('.')[0])
            ids.append(name)

        # Set super parameters
        TxtCorpusReader.__init__(self, name='ScisummNet-2019', path=Path('data/ScisummNet-2019_singles'), ids=ids,
                                 stop_words = stopwords.words('english'))

class CPMData(TxtCorpusReader):
    """
    Class for CPM data. Inherits from TxtCorpusReader,
    with a few corpus specific variables and methods.
    """

    def __init__(self):
        """
        Initialize parameters
        """

        # Containers for document ids
        ids = []
        self.stem_ids = []
        self.report_ids = []
        self.abstract_ids = []

        # Search data folder for document ids, add to containers.
        for fn in Path("data/CPM_singles").iterdir():
            name = fn.stem
            nametype = name.split('_')[1]
            if nametype == 'report':
                self.report_ids.append(name.split('.')[0])
                self.stem_ids.append(name.split('_')[0])
            else:
                self.abstract_ids.append(name.split('.')[0])
            ids.append(name)

        # Set super parameters
        TxtCorpusReader.__init__(self, name='CPM', path=Path('data/CPM_singles'), ids=ids,
                                 stop_words = stopwords.words('english'))






class AANData(TxtCorpusReader):
    """
        Class for AAN corpus data. Inherits from TxtCorpusReader,
        with a few corpus specific variables and methods.
    """

    def __init__(self):
        """
        Initialize parameters
        """
        self.path = Path('data/AAN_singles')

        # Containers for document ids
        ids = []

        # Search data folder for document ids, add to containers.
        for fn in self.path.iterdir():
            name = fn.stem
            ids.append(name)

        # Set super parameters
        TxtCorpusReader.__init__(self, name='AAN', path=self.path, ids=ids,
                                 stop_words = stopwords.words('english'))





