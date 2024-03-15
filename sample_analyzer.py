import pandas as pd
import os
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt


def calc_sample_clonal_freqs(freqs: pd.DataFrame):
    unique_samples = freqs['sample'].unique()
    unique_clones = np.arange(1, freqs['cluster'].max() + 1)
    values = np.zeros((len(unique_clones), len(unique_samples)))
    rownames = unique_clones.astype(str)
    colnames = unique_samples

    for i, clone in enumerate(unique_clones):
        for j, sample in enumerate(unique_samples):
            clone_sample_freqs = freqs.loc[(freqs['cluster'] == clone) & (freqs['sample'] == sample), 'frac']
            if not clone_sample_freqs.empty:
                values[i, j] = clone_sample_freqs.iloc[0]
    #assert np.all((0 <= values) & (values <= 1.))
    #assert np.all((1 - .05 <= np.sum(values, axis=0)) & (np.sum(values, axis=0) <= 1 + .05))

    # Calculate Kullback-Leibler divergence of the sample’s clonal frequency distribution from the average distribution over all samples of a patientid
    # i.e. inter-tumor heterogeneity
    print("values",np.sum(values, axis=0))
    p = (values / np.sum(values, axis=0)).T

    aug = pd.DataFrame(p, columns=[f'w{i}' for i in range(1, p.shape[1] + 1)], index=unique_samples)
    aug.index.names = ['sample']
    # save averaged clonal frequency distributions per sample
    # TODO: add equation coefficients (hc etc) columns and join dataframes (wi means ith clone and value is 𝑝𝑖𝑗 is the normalized frequency)
    # intres.append(pd.DataFrame({'hc': hc, 'c': c, 'hu': hu, 'u': u, 'n': n}, index = unique_samples, columns = aug.columns))

    # results = pd.concat(intres, axis=0, ignore_index=False)
    # auf.to_csv("/home/aimaaral/dev/tumor-evolution-2023/heterogeneity/avg_cfd.csv",sep = '\t')
    return aug.fillna(0)

class DataAnalyzer:
    def __init__(self, models, files):
        self.models = models
        self.files = files

    def calc_all_clonal_freqs(self):
        intres = []
        augarr = []
        for file in self.files:
            print(f"processing '{file}'..")
            basename = os.path.basename(file)
            patientid = re.sub(f'^([^_]+\\d+)(_v2)?_vaf_(.*)_cellular_freqs\\.csv$', '\\1', basename)

            freqs = pd.read_csv(file, sep='\t')  # non utf start byte error encoding='ISO-8859-1'
            freqs = freqs.loc[
                    freqs['model.num'] ==
                    self.models.loc[self.models['patient'].str.contains(patientid), 'model'].values[0], :]

            unique_samples = freqs['sample.id'].unique()
            unique_clones = np.arange(1, freqs['cloneID'].max() + 1)
            values = np.zeros((len(unique_clones), len(unique_samples)))
            rownames = unique_clones.astype(str)
            colnames = unique_samples

            for i, clone in enumerate(unique_clones):
                for j, sample in enumerate(unique_samples):
                    clone_sample_freqs = freqs.loc[
                        (freqs['cloneID'] == clone) & (freqs['sample.id'] == sample), 'cell.freq']
                    if not clone_sample_freqs.empty:
                        values[i, j] = clone_sample_freqs.iloc[0] / 100.
            assert np.all((0 <= values) & (values <= 1.))
            assert np.all((1 - .05 <= np.sum(values, axis=0)) & (np.sum(values, axis=0) <= 1 + .05))

            # Calculate Kullback-Leibler divergence of the sample’s clonal frequency distribution
            # from average distribution over all samples of a patientid i.e. inter-tumor heterogeneity
            p = (values / np.sum(values, axis=0)).T
            z = p * np.log(p)
            # Nan to 0
            z[~(p > 0.)] = 0.
            hc = -np.sum(z, axis=0)
            # Clonal complexity (latter sum)
            c = np.exp(hc)

            # first sum
            q = np.mean(p, axis=1)
            z = p * np.log(q.reshape(-1, 1))
            z[~(p > 0.)] = 0.
            hu = -np.sum(z, axis=1)
            u = np.exp(hu)
            # sum over rows (clones)
            n = np.sum(p > 0., axis=1)

            aug = pd.DataFrame(p, columns=[f'w{i}' for i in range(1, p.shape[1] + 1)], index=unique_samples)
            aug.index.names = ['sample']
            augarr.append([aug, c, u, n])
            # save averaged clonal frequency distributions per sample
            # TODO: add equation coefficients (hc etc) columns and join dataframes (wi means ith clone and value is 𝑝𝑖𝑗 is the normalized frequency)
            # intres.append(pd.DataFrame({'hc': hc, 'c': c, 'hu': hu, 'u': u, 'n': n}, index = unique_samples, columns = aug.columns))
        auf = pd.concat(augarr, axis=0, ignore_index=False).fillna(0)

        return auf


def calc_corr_matrix(cell_freqs, pid, plot=False):
    sns.set_theme(style="white")

    # Compute the correlation matrix
    corr = cell_freqs.T.corr()
    # print(d.T)

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    if plot:
        try:
            hmap = sns.heatmap(corr, cmap=cmap, center=0,
                               square=True, linewidths=.5, cbar_kws={"shrink": .5})
            plt.savefig("./plots/"+pid+"_heatmap.jpg")
        except Exception as e:
            print(e)
            pass
    # set upper triangle to zero
    corr *= np.tri(*corr.shape)
    return corr

from sklearn.datasets import load_digits
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import time
import warnings
from itertools import cycle, islice


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:

            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    # Plot the corresponding dendrogram
    print('linkage_matrix',linkage_matrix)
    dendrogram(linkage_matrix, leaf_font_size=16, **kwargs)


def hierarcical_clustering(cfds: pd.DataFrame, patient, n_clusters=None, distance_threshold=2, plot=False):

    plot_num = 1
    # plt.figure().set_figheight(30)

    # Compute the correlation matrix

    # d2 = d.set_index("sample")
    clusters = dict()

    embedding = SpectralEmbedding(n_components=3)
    if len(cfds) > 2:
        # Cluster all samples in patient
        # print(g)
        # X_transformed = embedding.fit_transform(X)
        # X_transformed.shape
        # clustering = AgglomerativeClustering().fit(X_transformed)
        if n_clusters:
            agc = AgglomerativeClustering(distance_threshold=None, n_clusters=n_clusters, compute_distances=True, compute_full_tree=True)
        else:
            agc = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None, compute_distances=True, compute_full_tree=True)

        agg = agc.fit(cfds)
        print("AgglomerativeClustering",cfds,agg.labels_)
        for index, l in enumerate(agg.labels_):
            clusters[cfds.index[index]] = l

        if plot:
            plt.rc_context({'lines.linewidth': 3.0})

            plt.figure().set_figwidth(15)
            plt.title("Hierarchical Clustering of Samples in patient " + patient, fontsize=20)
            # plot the top three levels of the dendrogram
            # plot_dendrogram(agg, truncate_mode="level", p=3, labels=group.index.str.split('_').str[1])
            plot_dendrogram(agg, labels=cfds.index)
            plt.savefig("./plots/"+patient + "_hclustering.png")
            plt.show()

            # plt.xlabel("Sample name")

        # TODO group by tissue and do hierarcical clustering per tissue
        # gr = pd.DataFrame(cfds)
        # print(gr.index)
        # gr['tissue'] = gr.index.str[1:3]
        #
        # for tissue_name, tissue_grp in gr.groupby('tissue'):
        #     # Cluster by tissue site
        #     # print(grp)
        #     if len(tissue_grp) > 2:
        #         gt = tissue_grp.drop(columns=["tissue"])
        #         aggt = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(gt)
        #         plt.rc_context({'lines.linewidth': 3.0})
        #
        #         plt.figure().set_figwidth(10)
        #
        #         plt.title("Hierarchical Clustering of Samples in tissue " + tissue_name, fontsize=20)
        #         # plot the top three levels of the dendrogram
        #         # plot_dendrogram(agg, truncate_mode="level", p=3, labels=group.index.str.split('_').str[1])
        #         plot_dendrogram(aggt, labels=gt.index)
        #         # plt.xlabel("Sample name")
        #         plt.savefig("./plots/hc_"+patient+"_"+tissue_name+".png")
        #         plt.show()

    return clusters

