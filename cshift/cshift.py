import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .utils import aggregate_chisquare_test, false_discovery_rate, percent_change


class CShift:
    """
    A method of performing cluster enrichments/depletions
    """

    def __init__(self, clusters: np.ndarray, groups: np.ndarray, reference: np.ndarray):
        self.clusters = np.array(clusters)
        self.groups = np.array(groups)
        self.reference = np.array(reference)

        self._validate_inputs()
        self._group_counts()
        self._cluster_counts()
        self._build_distributions()
        self._get_reference_idx()

    def _validate_inputs(self):
        assert (
            self.clusters.size == self.groups.size
        ), "clusters and groups must be the same size"
        assert np.isin(
            self.reference, self.groups
        ).all(), "reference must be a subset of groups"

    def _group_counts(self):
        self.g_unique, self.g_counts = np.unique(self.groups, return_counts=True)
        self.g_size = self.g_unique.size

    def _cluster_counts(self):
        self.c_unique, self.c_counts = np.unique(self.clusters, return_counts=True)
        self.c_size = self.c_unique.size

    def _build_distributions(self):
        self.distributions = np.zeros((self.g_unique.size, self.c_unique.size))
        for idx, g in enumerate(self.g_unique):
            for jdx, c in enumerate(self.c_unique):
                self.distributions[idx, jdx] = np.sum(
                    np.logical_and(self.clusters == c, self.groups == g)
                )

    def _get_reference_idx(self):
        self.ref_idx = np.flatnonzero(np.isin(self.g_unique, self.reference))

    def fit(self):
        """
        Performs the cluster shift enrichment analysis
        """
        self.pval_matrix = np.stack(
            [
                aggregate_chisquare_test(
                    self.distributions[self.ref_idx], self.distributions[i]
                )
                for i in np.arange(self.g_size)
            ]
        )
        self.pcc_matrix = np.stack(
            [
                percent_change(
                    self.distributions[self.ref_idx].mean(axis=0), self.distributions[i]
                )
                for i in np.arange(self.g_size)
            ]
        )
        self.qval_matrix = false_discovery_rate(self.pval_matrix)
        return (self.pcc_matrix, self.qval_matrix)

    def plot(
        self,
        transpose=False,
        threshold=0.05,
        filter_significant=True,
        center=0,
        linewidth=1.0,
        linecolor="black",
        show=True,
        **kwargs
    ):
        """
        Plot the cluster shift enrichment as a clustermap
        """
        mat = -np.log(self.qval_matrix) * np.sign(self.pcc_matrix)
        df = pd.DataFrame(mat, index=self.g_unique, columns=self.c_unique)
        if filter_significant:
            df = df.loc[(self.qval_matrix <= threshold).any(axis=1)]
            if df.empty:
                raise ValueError("No significant clusters found")

        if transpose:
            df = df.T

        fig = sns.clustermap(
            df,
            cmap="seismic",
            center=center,
            linecolor=linecolor,
            linewidth=linewidth,
            **kwargs,
        )
        if show:
            plt.show()
        else:
            return fig
