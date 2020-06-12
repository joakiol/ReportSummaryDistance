from TFIDFModel import TFIDFModel
from LSAModel import LSAModel
from LDAModel import LDAModel
from Doc2vecModel import Doc2vecModel
from Word2vecModel import Word2vecModel
from BertModel import BertModel

from data import ScisummData, AANData, CPMData, VenduData
from analysis import ScisummAnalysis, CPMAnalysis, VenduAnalysis, plot_correlation
import pandas as pd
from pathlib import Path

# Configurations for TF-IDF for each of the datasets
def TFIDF(analysis):

    model = TFIDFModel()

    if analysis == 'SSN':
        data = ScisummData()
        model.set_dict(data, no_below=5, remove_stopwords=True)
        analysis = ScisummAnalysis(model)
        acc = analysis.get_accuracy()
        analysis.plot_tSNE_with_lines(name='TFIDF_scatter.pdf', label=acc)
        analysis.plot_densities(name='TFIDF_Scisumm_density.pdf', label=True)

    elif analysis == 'CPM/AAN':
        print("TFIDF can't be trained on the AAN corpus. Use 'CPM/CPM' instead. ")

    elif analysis == 'CPM/CPM':
        data = CPMData()
        model.set_dict(data, no_below=5, remove_stopwords=True)
        analysis = CPMAnalysis(model)
        analysis.get_cross_validation_scores()
        analysis.plot_densities(name='tfidf_CPM_density.pdf', lab=True, axis=False, bigtext=False)

    elif analysis == 'Vendu':
        data = VenduData()
        model.set_dict(data, no_below=20, remove_stopwords=True)
        analysis = VenduAnalysis(model)
        analysis.plot_densities(name='TFIDF_density_vendu.pdf', lab=True)

    else:
        print("Invalid argument for model. Must be 'SSN', 'CPM/AAN', 'CPM/CPM' or 'Vendu'.")


# Configurations for LSA for each of the datasets
def LSA(analysis):

    model = LSAModel(vector_length=100)

    if analysis == 'SSN':
        data = ScisummData()
        model.set_dict(data, no_below=5, remove_stopwords=True)
        model.train(data)
        analysis = ScisummAnalysis(model)
        acc = analysis.get_accuracy()
        analysis.plot_tSNE_with_lines(name='LSA_scatter.pdf', label=acc)
        analysis.plot_densities(name='LSA_Scisumm_density.pdf')

    elif analysis == 'CPM/AAN':
        data = AANData()
        model.set_dict(data, no_below=20, remove_stopwords=True)
        model.train(data)
        analysis = CPMAnalysis(model)
        analysis.get_cross_validation_scores()
        analysis.plot_densities(name='LSA_CPM_density.pdf', lab=True)

    elif analysis == 'CPM/CPM':
        data = CPMData()
        model.set_dict(data, no_below=5, remove_stopwords=True)
        model.train(data)
        analysis = CPMAnalysis(model)
        analysis.get_cross_validation_scores()

    elif analysis == 'Vendu':
        data = VenduData()
        model.set_dict(data, no_below=20, remove_stopwords=True)
        model.train(data)
        analysis = VenduAnalysis(model)
        analysis.plot_densities(name='LSA_density_vendu.pdf')

    else:
        print("Invalid argument for model. Must be 'SSN', 'CPM/AAN', 'CPM/CPM' or 'Vendu'.")


# Configurations for LDA for each of the datasets
def LDA(analysis):

    model = LDAModel(vector_length=100)

    if analysis == 'SSN':
        data = ScisummData()
        model.set_dict(data, no_below=5, remove_stopwords=True)
        model.train(data, passes=30)
        analysis = ScisummAnalysis(model)
        acc = analysis.get_accuracy()
        analysis.plot_tSNE_with_lines(name='LDA_scatter.pdf', label=acc)
        analysis.plot_densities(name='LDA_Scisumm_density.pdf', axis=False)

    elif analysis == 'CPM/AAN':
        data = AANData()
        model.set_dict(data, no_below=20, remove_stopwords=True)
        model.train(data, passes = 10)
        analysis = CPMAnalysis(model)
        analysis.get_cross_validation_scores()
        analysis.plot_densities(name='LSA_CPM_density.pdf', lab=True)

    elif analysis == 'CPM/CPM':
        data = CPMData()
        model.set_dict(data, no_below=5, remove_stopwords=True)
        model.train(data, passes=20)
        analysis = CPMAnalysis(model)
        analysis.get_cross_validation_scores()

    elif analysis == 'Vendu':
        data = VenduData()
        model.set_dict(data, no_below=20, remove_stopwords=True)
        model.train(data, passes=10)
        analysis = VenduAnalysis(model)
        analysis.plot_densities(name='LDA_density_vendu.pdf', axis=False)

    else:
        print("Invalid argument for model. Must be 'SSN', 'CPM/AAN', 'CPM/CPM' or 'Vendu'.")


# Configurations for Word2vec for each of the datasets
def Word2vec(analysis):

    if analysis == 'SSN':
        data = ScisummData()
        model = Word2vecModel(vector_length=100, type='SG', window=6, min_count=5)
        model.train([data], epochs=100)
        analysis = ScisummAnalysis(model)
        acc = analysis.get_accuracy()
        analysis.plot_tSNE_with_lines(name='Word2vec_scatter.pdf', label=acc)
        analysis.plot_densities(name='Word2vec_Scisumm_density.pdf')

    elif analysis == 'CPM/AAN':
        data = AANData()
        model = Word2vecModel(vector_length=100, type='SG', window=6, min_count=20)
        model.train([data], epochs=10)
        analysis = CPMAnalysis(model)
        analysis.get_cross_validation_scores()
        analysis.plot_densities(name='Word2vec_CPM_density.pdf')

    elif analysis == 'CPM/CPM':
        data = CPMData()
        model = Word2vecModel(vector_length=100, type='SG', window=6, min_count=5)
        model.train([data], epochs=100)
        analysis = CPMAnalysis(model)
        analysis.get_cross_validation_scores()

    elif analysis == 'Vendu':
        data = VenduData()
        model = Word2vecModel(vector_length=100, type='SG', window=6, min_count=20)
        model.train([data], epochs=10)
        analysis = VenduAnalysis(model)
        analysis.plot_densities(name='Word2vec_density_vendu.pdf')

    else:
        print("Invalid argument for model. Must be 'SSN', 'CPM/AAN', 'CPM/CPM' or 'Vendu'.")


# Configurations for Doc2vec for each of the datasets
def Doc2vec(analysis):

    if analysis == 'SSN':
        data = ScisummData()
        model = Doc2vecModel(vector_length=100, type='DBOW', window=6, min_count=5)
        model.train([data], epochs=100)
        analysis = ScisummAnalysis(model)
        acc = analysis.get_accuracy()
        analysis.plot_tSNE_with_lines(name='Doc2vec_scatter.pdf', label=acc)
        analysis.plot_densities(name='Doc2vec_Scisumm_density.pdf')

    elif analysis == 'CPM/AAN':
        data = AANData()
        model = Doc2vecModel(vector_length=100, type='DBOW', window=6, min_count=20)
        model.train([data], epochs=10, infer=True)
        analysis = CPMAnalysis(model)
        analysis.get_cross_validation_scores()
        analysis.plot_densities(name='Doc2vec_CPM_density.pdf')

    elif analysis == 'CPM/CPM':
        data = CPMData()
        model = Doc2vecModel(vector_length=100, type='DBOW', window=6, min_count=5)
        model.train([data], epochs=100)
        analysis = CPMAnalysis(model)
        analysis.get_cross_validation_scores()

    elif analysis == 'Vendu':
        data = VenduData()
        model = Doc2vecModel(vector_length=100, type='DBOW', window=6, min_count=20)
        model.train([data], epochs=10)
        analysis = VenduAnalysis(model)
        analysis.plot_densities(name='Doc2vec_density_vendu.pdf')

    else:
        print("Invalid argument for model. Must be 'SSN', 'CPM/AAN', 'CPM/CPM' or 'Vendu'.")

# Configurations for S-BERT for each of the datasets
def SBERT(analysis):

    model = BertModel('bert-base-nli-stsb-mean-tokens')

    if analysis == 'SSN':
        analysis = ScisummAnalysis(model)
        acc = analysis.get_accuracy()
        analysis.plot_tSNE_with_lines(name='BERT_scatter.pdf', label=acc)
        analysis.plot_densities(name='BERT_Scisumm_density.pdf')

    elif analysis == 'CPM/AAN':
        analysis = CPMAnalysis(model)
        analysis.get_cross_validation_scores()
        analysis.plot_densities(name='BERT_CPM_density.pdf')

    elif analysis == 'CPM/CPM':
        print("SBERT can't be trained on the CPM data. Use CPM/AAN instead.")

    elif analysis == 'Vendu':
        print("SBERT can't be trained on Vendu.")

    else:
        print("Invalid argument for model. Must be 'SSN', 'CPM/AAN', 'CPM/CPM' or 'Vendu'.")


# Correlation plot configuration
def correlation_Vendu():

    data = VenduData()

    TFIDFmodel = TFIDFModel()
    TFIDFmodel.set_dict(data, no_below=20, remove_stopwords=True)
    TFIDFanalysis = VenduAnalysis(TFIDFmodel)

    LSAmodel = LSAModel(vector_length=100)
    LSAmodel.set_dict(data, no_below=20, remove_stopwords=True)
    LSAmodel.train(data)
    LSAanalysis = VenduAnalysis(LSAmodel)

    W2Vmodel = Word2vecModel(vector_length=100, type='SG', window=6, min_count=20)
    W2Vmodel.train([data], epochs=10)
    W2Vanalysis = VenduAnalysis(W2Vmodel)

    D2Vmodel = Doc2vecModel(vector_length=100, type='DBOW', window=6, min_count=20)
    D2Vmodel.train([data], epochs=10)
    D2Vanalysis = VenduAnalysis(D2Vmodel)

    distances = pd.DataFrame({'TFIDF' : TFIDFanalysis.distances['own'],
                              'LSA' : LSAanalysis.distances['own'],
                              'Word2vec': W2Vanalysis.distances['own'],
                              'Doc2vec': D2Vanalysis.distances['own']})

    plot_correlation(distances)

if __name__ == '__main__':

    # Plot folder must be made (This can be deleted after first run)
    Path('plot').mkdir(parents=True, exist_ok=True)

    # Which analysis should be performed? 'SSN', 'CPM/AAN', 'CPM/CPM'.
    # 'Vendu' can't be run since the data will not be uploaded to GitHub.
    analysis = 'SSN'

    # Uncomment models to analyse. Different hyperparameters can be run by changing above functions.
    TFIDF(analysis)
    LSA(analysis)
    LDA(analysis)
    Word2vec(analysis)
    Doc2vec(analysis)
    SBERT(analysis)

    # Makes correlation plot for Vendu data. This can't be run, since Vendu data will not be uploaded to github.
    #correlation_Vendu()