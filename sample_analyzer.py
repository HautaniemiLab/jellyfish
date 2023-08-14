import pandas as pd
import os
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

class DataAnalyzer:
    def __init__(self, models, files):
        self.models = models
        self.files = files

    def calc_sample_clonal_freqs(self, freqs: pd.DataFrame):

        unique_samples = freqs['sample'].unique()
        unique_clones = np.arange(1, freqs['cluster'].max() + 1)
        values = np.zeros((len(unique_clones), len(unique_samples)))
        rownames = unique_clones.astype(str)
        colnames = unique_samples

        for i, clone in enumerate(unique_clones):
            for j, sample in enumerate(unique_samples):
                clone_sample_freqs = freqs.loc[(freqs['cluster'] == clone) & (freqs['sample'] == sample), 'freq']
                if not clone_sample_freqs.empty:
                    values[i, j] = clone_sample_freqs.iloc[0]
        assert np.all((0 <= values) & (values <= 1.))
        assert np.all((1 - .05 <= np.sum(values, axis=0)) & (np.sum(values, axis=0) <= 1 + .05))

        # Calculate Kullback-Leibler divergence of the sample’s clonal frequency distribution from the average distribution over all samples of a patientid
        # i.e. inter-tumor heterogeneity
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
        # save averaged clonal frequency distributions per sample
        # TODO: add equation coefficients (hc etc) columns and join dataframes (wi means ith clone and value is 𝑝𝑖𝑗 is the normalized frequency)
        # intres.append(pd.DataFrame({'hc': hc, 'c': c, 'hu': hu, 'u': u, 'n': n}, index = unique_samples, columns = aug.columns))

        # results = pd.concat(intres, axis=0, ignore_index=False)
        # auf.to_csv("/home/aimaaral/dev/tumor-evolution-2023/heterogeneity/avg_cfd.csv",sep = '\t')
        return aug.fillna(0)

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

            # Calculate Kullback-Leibler divergence of the sample’s clonal frequency distribution from the average distribution over all samples of a patientid
            # i.e. inter-tumor heterogeneity
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
            augarr.append(aug)
            # save averaged clonal frequency distributions per sample
            # TODO: add equation coefficients (hc etc) columns and join dataframes (wi means ith clone and value is 𝑝𝑖𝑗 is the normalized frequency)
            # intres.append(pd.DataFrame({'hc': hc, 'c': c, 'hu': hu, 'u': u, 'n': n}, index = unique_samples, columns = aug.columns))
        auf = pd.concat(augarr, axis=0, ignore_index=False).fillna(0)
        # results = pd.concat(intres, axis=0, ignore_index=False)
        # auf.to_csv("/home/aimaaral/dev/tumor-evolution-2023/heterogeneity/avg_cfd.csv",sep = '\t')
        return auf


def calc_corr_matrix(cell_freqs):
    sns.set_theme(style="white")

    # Generate a large random dataset
    # d = pd.read_csv('/home/aimaaral/dev/tumor-evolution-2023/heterogeneity/cellular_freqs.tsv',sep = '\t').set_index("sample").drop(columns=["hc","hu","c","u","n"])

    # TODO: group by patient and generate corrmatrix for each patient separately
    # Compute the correlation matrix
    corr = cell_freqs.T.corr()
    # print(d.T)

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    try:
        hmap = sns.heatmap(corr, cmap=cmap, center=0,
                           square=True, linewidths=.5, cbar_kws={"shrink": .5})
        hmap
    except Exception as e:
        print(e)
        pass
    return corr
