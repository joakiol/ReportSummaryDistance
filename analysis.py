import numpy as np
import pandas as pd
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from data import ScisummData, CPMData, VenduData
from scipy.spatial.distance import cdist
from pathlib import Path
import pickle
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine

class VenduAnalysis:
    """
        Class for Vendu analysis, with useful functions
    """
    def __init__(self, model, distance_measure = 'cosine'):
        """
            Initialize analysis. Calculate distances.

            :param model: Model to analyse
        """

        self.distance_measure = distance_measure
        self.data = VenduData()
        self.model = model
        self.model.fit(self.data)
        try:
            self.distances = pd.read_csv(self.model.path / str('distances_%s_%s.csv' %
                                                               (self.data.name, self.distance_measure)), index_col=0)
        except:
            self.distances = self.calculate_distances()

    def calculate_distances(self):
        """
        Calculates distance from reports to their own summary, as well as to a random summary.

        :return: Panda dataframe of distances.
        """

        # Initialize container.
        distances = np.zeros((len(self.data.stem_ids), 2))

        # For each report-abstract pairs
        for i in tqdm(range(len(self.data.stem_ids))):

            # Get report, abstract and random other abstract
            report = self.model.doc_vecs.loc['%s_report' % self.data.stem_ids[i]]
            summary = self.model.doc_vecs.loc['%s_abstract' % self.data.stem_ids[i]]
            other = self.model.doc_vecs.loc[self.data.abstract_ids[random.randint(0, len(self.data.abstract_ids)-1)]]

            # self.distance_measure is always cosine. Calculate distance.
            if self.distance_measure == 'cosine':
                distances[i][0] = cosine(report, summary)
                distances[i][1] = cosine(report, other)

        # Make pandas dataframe, save and return.
        distances = pd.DataFrame(distances, index=self.data.stem_ids, columns=['own', 'other'])
        distances.to_csv(self.model.path / str('distances_%s_%s.csv' % (self.data.name, self.distance_measure)))

        return distances


    def plot_densities(self, name=None, lab=False, axis = True):
        """
        Plot two densities, one for distance to own report, the other for distance to other reports.
        """

        true_samples = self.distances['own']
        true_samples = true_samples[~np.isnan(true_samples)]
        false_samples = self.distances['other']
        false_samples = false_samples[~np.isnan(false_samples)]

        # Plot histogram
        plt.figure(figsize=(8,5))
        plt.rcParams.update({'font.size': 22})
        plt.hist(true_samples, bins=100, density=True, facecolor='g', alpha=0.5, label="Distance to own")
        plt.hist(false_samples, bins=100, density=True, facecolor='r', alpha=0.5, label="Distance to other")

        plt.title(self.model.shortname)
        if axis:
            plt.xlim(0,1)
        plt.xlabel('Distance')
        plt.ylabel('Density')
        if lab:
            plt.legend(loc='upper left')

        plt.tight_layout()

        if name != None:
            plt.savefig(str(Path('plot/%s' % name)))
        plt.show()


class CPMAnalysis:
    """
        Class for CPM analysis, with useful functions
    """

    def __init__(self, model, distance_measure = 'cosine'):
        """
            Initialize analysis. Calculate distances.

            :param model: Model to analyse
        """

        self.distance_measure = distance_measure
        self.data = CPMData()
        self.model = model
        self.model.fit(self.data)
        self.distances = self.calculate_distances()
        self.distances = self.distances.loc[self.data.stem_ids,0].to_numpy()
        self.labels = pd.read_csv(Path('data/CPM_labels.csv'), index_col=0)
        self.labels = self.labels.iloc[[int(id) for id in self.data.stem_ids],0].to_numpy()

    def plot_densities(self, name=None, lab=False, axis=True, bigtext=True):
        """
        Plot two densities, one for distance to own report, the other for distance to other reports.
        """

        true_samples = np.array([self.distances[i] for i in range(len(self.distances)) if self.labels[i] == 1])
        false_samples = np.array([self.distances[i] for i in range(len(self.distances)) if self.labels[i] == 0])


        # Plot histogram
        plt.figure(figsize=(8,5))
        plt.rcParams.update({'font.size': 15})
        if bigtext:
            plt.rcParams.update({'font.size': 22})
        plt.hist(true_samples, bins=30, density=True, facecolor='g', alpha=0.5, label="Good match")
        plt.hist(false_samples, bins=30, density=True, facecolor='r', alpha=0.5, label="Bad match")

        plt.title(self.model.shortname)
        if axis:
            plt.xlim(0,1)
        plt.xlabel('Distance')
        plt.ylabel('Density')
        if lab:
            plt.legend(loc='upper left')

        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9, wspace=0, hspace=0)

        if name != None:
            plt.savefig(str(Path('plot/%s' % name)))
        plt.show()


    def calculate_distances(self):
        """
        Function that calculates the distance from all reports to all abstracts in Scisumm-data.
        """

        # Matrices with reports vectors and abstracts vectors
        reports = self.model.doc_vecs.loc[self.data.report_ids]
        abstracts = self.model.doc_vecs.loc[self.data.abstract_ids]


        # Calculates the distance between each pairs of the matrices
        distances = cdist(reports, abstracts, self.distance_measure)
        distances = np.nan_to_num(distances, nan=np.inf)
        distances = np.diagonal(distances)

        distances = pd.DataFrame(distances, index=self.data.stem_ids)

        return distances

    def get_pred(self, threshold, distances=None):
        """
        Get CPM predictions based given threshold

        :param threshold: Predict values lower than threshold as 1 (good match), while 0 (bad match) for values above
        :param distances: Document distances can be given as input. If not, uses self.distances.
        :return: List of 0/1, predictions for the distances in input.
        """

        if type(distances) == type(None):
            distances = self.distances

        result = []

        for i in range(len(distances)):
            if distances[i] <= threshold:
                result.append(1)
            else:
                result.append(0)

        return result


    def get_scores_given_pred(self, pred, true):
        """
        Returns scores based on input list of predicted and true labels.

        :param pred: Predicted labels.
        :param true: True labels
        :return: accuracy, precision, recall, f_one
        """


        accuracy = sum(pred == true) / len(pred)
        precision, recall, f_one, support = precision_recall_fscore_support(true, pred, average='binary')

        return accuracy, precision, recall, f_one

    def get_scores_given_threshold(self, threshold, distances=None, labels = None):
        """
        Get scores (acc, prec, rec, F1) given a threshold (and possibly distances/labels)

        :param threshold: Threshold to use between good/bad predictions
        :param distances: Distances to get scores on
        :param labels: True labels
        :return:
        """

        if type(distances) == type(None):
            distances = self.distances

        if type(labels) == type(None):
            labels = self.labels

        # Get predictions
        pred = self.get_pred(threshold, distances)

        true = labels

        # Get scores
        accuracy = sum(pred == true)/len(pred)
        precision, recall, f_one, support = precision_recall_fscore_support(true, pred, average='binary')

        return accuracy, precision, recall, f_one

    def get_scores_by_threshold(self, distances=None, labels=None):
        """
        Calculates scores as a function of threshold
        :param distances: Distances to predict, and calculate score of
        :param labels: True labels
        :return: List of accuracy, precision, recall, F1 as a function of threshold.
        """

        if type(distances) == type(None):
            distances = self.distances
        if type(labels) == type(None):
            labels = self.labels

        x = np.linspace(0, max(distances), 100)

        accuracies = []
        precisions = []
        recalls = []
        f_ones = []

        for trsh in x:
            accuracy, precision, recall, f_one = self.get_scores_given_threshold(trsh, distances, labels)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f_ones.append(f_one)

        return x, accuracies, precisions, recalls, f_ones

    def get_cross_validation_scores(self):
        """
        Performs K-fold cross-validation on data.
        :return: Average scores.
        """

        kf = KFold(n_splits=10)

        accuracy = 0
        precision = 0
        recall = 0
        f_one = 0

        # For each train, test
        for train_index, test_index in kf.split(self.distances):
            distances_train = self.distances[train_index]
            distances_test = self.distances[test_index]

            labels_train = self.labels[train_index]
            labels_test = self.labels[test_index]

            # Get scores as function of threshold on train data
            x, accuracies, precisions, recalls, f_ones = self.get_scores_by_threshold(distances=distances_train,
                                                                                    labels=labels_train)

            # Chose best threshold on train data
            threshold = x[np.argmax(accuracies)]

            # Use this threshold to predict test data
            pred = self.get_pred(threshold, distances=distances_test)

            # Get scores, add to container
            new_accuracy, new_precision, new_recall, new_f_one = self.get_scores_given_pred(pred, true=labels_test)

            accuracy += new_accuracy
            precision += new_precision
            recall += new_recall
            f_one += new_f_one


        # Print resulting average scores.
        print('Accuracy \t= %.3f' % (accuracy/10))
        print('Precision \t= %.3f' % (precision/10))
        print('Recall \t= %.3f' % (recall/10))
        print('F1 \t\t= %.3f' % (f_one/10))



class ScisummAnalysis:
    """
    Class for ScisummNet-2019 analysis, with useful functions
    """

    def __init__(self, model, distance_measure = 'cosine'):
        """
        Initialize analysis. Calculate distances.

        :param model: Model to analyse
        """

        self.distance_measure = distance_measure
        self.data = ScisummData()
        self.model = model
        self.model.fit(self.data)
        self.distances = self.calculate_distances()

    def calculate_distances(self):
        """
        Function that calculates the distance from all reports to all abstracts in Scisumm-data.
        """

        # Matrices with reports vectors and abstracts vectors
        reports = self.model.doc_vecs.loc[self.data.report_ids]
        abstracts = self.model.doc_vecs.loc[self.data.abstract_ids]

        # Calculates the distance between each pairs of the matrices
        distances = cdist(reports, abstracts, self.distance_measure)
        distances = np.nan_to_num(distances, nan=np.inf)

        distances = pd.DataFrame(distances, index=self.data.report_ids, columns=self.data.abstract_ids)

        return distances


    def own_is_closest(self):
        """
        Get a list of True/False values that indicates whether an abstract has its own report as closest.

        :return: List of True/False
        """

        distances_np = self.distances.to_numpy()
        own_is_closest = np.argmin(distances_np, axis=0) == np.linspace(0, len(distances_np) - 1, len(distances_np))

        return pd.Series(data=own_is_closest, index = self.distances.columns)

    def get_accuracy(self):
        """
        Calculate Scisumm data accuracy for model.

        :return: Accuracy, as fraction of abstracts that have its own report as its closest.
        """

        own_is_closest = self.own_is_closest().values

        return np.sum(own_is_closest/len(own_is_closest))

    def plot_densities(self, name=False, label=False, axis=True):
        """
        Plot two densities, one for distance to own report, the other for distance to other reports.
        """

        # All distances
        distances_np = self.distances.copy().to_numpy()

        # Distance to own is on the diagonal
        own = np.diagonal(distances_np.copy())
        own = own[~np.isinf(own)]

        # Distance to others is everywhere but on the diagonal
        other = distances_np.copy()
        np.fill_diagonal(other, float('nan'))
        other = other.flatten()
        other = other[~np.isnan(other)]
        other = other[~np.isinf(other)]

        # Plot histogram

        plt.figure(figsize=(8, 5))
        plt.rcParams.update({'font.size': 22})

        plt.hist(own, bins=30, density=True, facecolor='g', alpha=0.5, label="Distance to own")
        plt.hist(other, bins=100, density=True, facecolor='r', alpha=0.5, label="Distance to other")

        if axis:
            plt.xlim(0,1)
        plt.xlabel('Distance')
        plt.ylabel('Density')
        plt.title(self.model.shortname)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9, wspace=0, hspace=0)

        if label:
            plt.legend(loc='upper left')
        if name != False:
            plt.savefig(str(Path('plot/%s' % name)))
        plt.show()

    def plot_quantiles(self, axis=0):
        """
        This is not included in report. Visualize distances by median and 0.95, 0.75, 0.25 and 0.05 quantiles

        :param axis:    Axis=0 shows distance from abstracts to reports.
                        Axis=1 shows distance from reports to abstracts
        """

        distances = self.distances.copy().to_numpy()

        own = np.diagonal(distances).copy()

        np.fill_diagonal(distances, 1)


        # Get quantiles
        top = np.quantile(distances, 0.95, axis=axis)
        upper = np.quantile(distances, 0.75, axis=axis)
        mean = np.quantile(distances, 0.5, axis=axis) - own
        lower = np.quantile(distances, 0.25, axis=axis)
        bottom = np.quantile(distances, 0.05, axis=axis)

        min = np.min(distances, axis=axis) - own

        second = -np.sort(-np.partition(distances, kth=1, axis=axis)[1] + own)
        third = -np.sort(-np.partition(distances, kth=2, axis=axis)[2] + own)
        fifth = -np.sort(-np.partition(distances, kth=4, axis=axis)[4] + own)
        tenth = -np.sort(-np.partition(distances, kth=9, axis=axis)[9] + own)




        indices = np.argsort(-min)
        indices2 = np.argsort(-mean)

        # Plot quantiles
        fig, ax = plt.subplots()
        x = np.linspace(1,len(distances), len(distances))

        ax.plot(x, mean[indices2], color='blue', label='Median')
        ax.plot(x, np.zeros(len(x)), color='green', label='Own')
        ax.plot(x, min[indices], color='red', label='Min')
        ax.plot(x, second, color='orange', label='Second')
        ax.plot(x, third, color='purple', label='Third')
        ax.plot(x, fifth, color='cyan', label='Fifth')
        ax.plot(x, tenth, color='lightblue', label='Tenth')


        if axis:
            plt.xlabel('Reports, sorted by median')
            plt.ylabel('Distance to abstracts')

        else:
            plt.xlabel('Abstracts, sorted by median')
            plt.ylabel('Distance to reports')

        plt.legend()
        plt.show()


    def plot_tSNE_with_lines(self, n=100, name=False, label=False):
        """
        Visualise topic/feature vectors from model, with a line from each report
        to its corresponding abstract. n reports will be chosen by random.

        :param n: Number of reports to plot. Maximum 1000.
        """

        # Get list of reports and abstracts, and random index list
        reports = self.distances.index.to_numpy()
        abstracts = self.distances.columns.to_numpy()
        indices = random.sample([i for i in range(len(reports))], n)

        # Get random subset of n reports and abstracts to plot
        reports_toplot = reports[indices]
        abstracts_toplot = abstracts[indices]

        # list with both
        all_toplot = np.concatenate((reports_toplot, abstracts_toplot))

        # Transform relevant data using t-SNE
        X = self.model.doc_vecs.loc[all_toplot]
        X_embedded = pd.DataFrame(TSNE(n_components=2).fit_transform(X), index=all_toplot)

        plt.figure(figsize=(8, 5))
        plt.rcParams.update({'font.size': 22})
        ax = plt.subplot(111)

        # Plot reports as blue
        ax.scatter(X_embedded.loc[reports_toplot,0], X_embedded.loc[reports_toplot,1],
                    color='blue', label='reports')

        # Plot abstracts as red
        ax.scatter(X_embedded.loc[abstracts_toplot, 0], X_embedded.loc[abstracts_toplot, 1],
                    color='orange', label='abstracts')

        # List with True/False for each abstract, True if its own report is closest
        list_own_is_closest = self.own_is_closest()

        # For each report-abstract pair
        for i in range(len(abstracts_toplot)):

            # Get coordinates
            temp = X_embedded.loc[[reports_toplot[i], abstracts_toplot[i]]]

            # If own report is closest neighbur to abstract, plot green line
            if list_own_is_closest.loc[abstracts_toplot[i]]:
                plt.plot(temp[0], temp[1], color='green', label='_nolegend_')

            # Else plot red line
            else:
                ax.plot(temp[0], temp[1], color='red', label='_nolegend_')

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0, hspace=0)
        if label:
            #box = ax.get_position()
            #ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
            #plt.legend(loc=(0.15,1.15), ncol=2)
            ax.legend([Line2D([0], [0], color='green')], [str('%i%%' % int(label*100))], framealpha=1, loc='upper left')
        plt.title(self.model.shortname)
        if name:
            plt.savefig('plot/%s' % name)
        plt.show()


# A function for creating correlation plot
def plot_correlation(dataframe):
    # Initialize figure
    fig, axs = plt.subplots(ncols=4, nrows=4, sharex='col', sharey='row', figsize=(8, 8))
    gridsize = 100

    # Cell 1, create histogram, adjust according to axes, plot and set labels.
    ax = axs[0, 0]
    TFIDF = dataframe['TFIDF']
    TFIDF = TFIDF[~np.isnan(TFIDF)]
    Values, Edges = np.histogram(TFIDF, bins=100)
    Edges = (Edges[1:] + Edges[:-1]) / 2
    Values = Values / max(Values) * 0.5 + 0.5
    ax.bar(Edges, Values, width=1 / (len(Edges) - 10), align='center', color='tab:blue')
    ax.set_ylabel('TF-IDF')
    ax.set_ylim([0.5, 1.01])

    # Cell 2, hexbin (2d-hist) plot between TF-IDF and LSA
    ax = axs[0, 1]
    ax.hexbin(dataframe['LSA'], dataframe['TFIDF'], gridsize=gridsize, cmap='Blues')

    # Cell 3
    ax = axs[0, 2]
    ax.hexbin(dataframe['Word2vec'], dataframe['TFIDF'], gridsize=gridsize, cmap='Blues')

    # Cell 4
    ax = axs[0, 3]
    ax.hexbin(dataframe['Doc2vec'], dataframe['TFIDF'], gridsize=gridsize, cmap='Blues')

    # Cell 5
    ax = axs[1, 0]
    corr = pearsonr(np.nan_to_num(dataframe['TFIDF']), np.nan_to_num(dataframe['LSA']))
    ax.text(0.61, 0.46, str(r'$r$ = %.2f' % corr[0]), fontsize=15)
    ax.set_ylabel('LSA')
    ax.set_ylim([-0.01, 1.01])

    # Cell 6
    ax = axs[1, 1]
    LSA = dataframe['LSA']
    LSA = LSA[~np.isnan(LSA)]
    Values, Edges = np.histogram(LSA, bins=100)
    Edges = (Edges[1:] + Edges[:-1]) / 2
    Values = Values / max(Values) * 1
    ax.bar(Edges, Values, width=1 / (len(Edges) - 10), align='center', color='tab:blue')

    # Cell 7
    ax = axs[1, 2]
    ax.hexbin(dataframe['Word2vec'], dataframe['LSA'], gridsize=gridsize, cmap='Blues')

    # Cell 8
    ax = axs[1, 3]
    ax.hexbin(dataframe['Doc2vec'], dataframe['LSA'], gridsize=gridsize, cmap='Blues')

    # Cell 9
    ax = axs[2, 0]
    corr = pearsonr(np.nan_to_num(dataframe['TFIDF']), np.nan_to_num(dataframe['Word2vec']))
    ax.text(0.61, 0.22, str(r'$r$ = %.2f' % corr[0]), fontsize=15)
    ax.set_ylabel('Word2vec')
    ax.set_ylim([-0.01, 0.5])

    # Cell 10
    ax = axs[2, 1]
    corr = pearsonr(np.nan_to_num(dataframe['LSA']), np.nan_to_num(dataframe['Word2vec']))
    ax.text(0.2, 0.22, str(r'$r$ = %.2f' % corr[0]), fontsize=15)

    # Cell 11
    ax = axs[2, 2]
    W2V = dataframe['Word2vec']
    W2V = W2V[~np.isnan(W2V)]
    Values, Edges = np.histogram(W2V, bins=100)
    Edges = (Edges[1:] + Edges[:-1]) / 2
    Values = Values / max(Values) * 0.5
    ax.bar(Edges, Values, width=1 / (len(Edges) - 10), align='center', color='tab:blue')

    # Cell 12
    ax = axs[2, 3]
    ax.hexbin(dataframe['Doc2vec'], dataframe['Word2vec'], gridsize=gridsize, cmap='Blues')

    # Cell 13
    ax = axs[3, 0]
    corr = pearsonr(np.nan_to_num(dataframe['TFIDF']), np.nan_to_num(dataframe['Doc2vec']))
    ax.text(0.61, 0.46, str(r'$r$ = %.2f' % corr[0]), fontsize=15)
    ax.set_xlabel('TF-IDF')
    ax.set_ylabel('Doc2vec')
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlim([0.5, 1.01])

    # Cell 14
    ax = axs[3, 1]
    corr = pearsonr(np.nan_to_num(dataframe['LSA']), np.nan_to_num(dataframe['Doc2vec']))
    ax.text(0.2, 0.46, str(r'$r$ = %.2f' % corr[0]), fontsize=15)
    ax.set_xlabel('LSA')
    ax.set_xlim([-0.01, 1.01])

    # Cell 15
    ax = axs[3, 2]
    corr = pearsonr(np.nan_to_num(dataframe['Word2vec']), np.nan_to_num(dataframe['Doc2vec']))
    ax.text(0.1, 0.46, str(r'$r$ = %.2f' % corr[0]), fontsize=15)
    ax.set_xlabel('Word2vec')
    ax.set_xlim([-0.01, 0.5])

    # Cell 16
    ax = axs[3, 3]
    Values, Edges = np.histogram(dataframe['Doc2vec'], bins=100)
    Edges = (Edges[1:] + Edges[:-1]) / 2
    Values = Values / max(Values) * 1
    ax.bar(Edges, Values, width=1 / (len(Edges) - 20), align='center', color='tab:blue')
    ax.set_xlim([-0.01, 1])
    ax.set_xlabel('Doc2vec')

    # Overall stuff, plot and save
    plt.rcParams.update({'font.size': 26})
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.15, wspace=0.15)
    plt.savefig('plot/correlation.pdf')
    plt.show()


