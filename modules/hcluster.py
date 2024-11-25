import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram

def spectra_similarity(spectra1, spectra2):
    spectra1 = np.array(spectra1)
    spectra2 = np.array(spectra2)
        # if nans at some position in any of the spectra, ignore that position
    mask = np.logical_and(np.isfinite(spectra1), np.isfinite(spectra2))
    spectra1 = spectra1[mask]
    spectra2 = spectra2[mask]
    return np.sum((spectra1 - spectra2)**2)

def spectra_similarity_corr(spectrum1, spectrum2, N=10):
    spectrum1 = np.array(spectrum1)
    spectrum2 = np.array(spectrum2)
    mask = np.logical_and(np.isfinite(spectrum1), np.isfinite(spectrum2))
    spectrum1 = spectrum1[mask]
    spectrum2 = spectrum2[mask]
    L = len(spectrum1)
    shifts = np.arange(-N, N + 1)
    num_shifts = len(shifts)
    
    # Pad spectrum1 to accommodate shifts without changing length
    padded_spectrum1 = np.pad(spectrum1, (N, N), mode='constant', constant_values=0)
    
    # Generate indices for shifted versions of spectrum1
    idx_shifts = shifts + N  # Adjust shifts due to padding
    indices = idx_shifts[:, None] + np.arange(L)
    
    # Create shifted versions of spectrum1
    shifted_spectrum1 = padded_spectrum1[indices]
    
    # Create mask to exclude comparisons with zero-padded elements
    col_indices = np.arange(L)
    shift_matrix = shifts[:, None]
    mask = np.where(shift_matrix >= 0,
                    col_indices >= shift_matrix,
                    col_indices < L + shift_matrix)
    
    # Compute differences and apply mask
    diffs = shifted_spectrum1 - spectrum2
    diffs[~mask] = 0  # Zero out non-overlapping regions
    
    # Compute SSD for each shift
    ssd = np.sum(diffs ** 2, axis=1)
    
    # Return the minimum SSD
    return np.min(ssd)

class hcluster:

    def __init__(self, cube, metric=None, n=12):
        self.cube = cube # cube object
        self.data = cube.get_cube(normalize=True)
        self.shape = self.data.shape
        self.data = self.data.reshape(self.shape[0], -1).T
        self.set_clusters(n) # number of clusters
        self.set_metric(metric)

    def set_clusters(self, n):
        self.n = n

    def set_metric(self, metric):
        self.metric = metric
        if self.metric is None:
            self.metric = spectra_similarity
    
    def compute_distances(self):
        self.distances = pdist(self.data, metric=self.metric)

    def compute_linkage(self):
        self.linkage = linkage(self.distances, method='ward')

    def compute_clusters(self, n=None):
        if n is not None:
            self.set_clusters(n)
        self.cluster_labels = fcluster(self.linkage, self.n, criterion='maxclust')

    def compute(self, n=None):
        self.compute_distances()
        self.compute_linkage()
        self.compute_clusters(n)

    def get_clusters(self, matrix=False):
        return self.cluster_labels if not matrix else self.cluster_labels.reshape(self.shape[1:])

    def compare(self, x1, y1, x2, y2):
        ''' Compare two pixels '''
        lambdas = self.cube.lambdas
        spectrum1 = self.cube.get_spectrum(x1, y1)
        spectrum2 = self.cube.get_spectrum(x2, y2)
        metric = self.metric(spectrum1, spectrum2)
        return lambdas, spectrum1, spectrum2, metric

    def get_spectra_in_cluster(self, cluster):
        ''' Get the spectra in a cluster '''
        mask = self.cluster_labels == cluster
        return self.data[mask]

    def get_metrics_in_cluster(self, cluster):
        ''' Get the metrics in a cluster '''
        spectra = self.get_spectra_in_cluster(cluster)
        # Mean
        mean_spectrum = np.mean(spectra, axis=0)
        # metric wrt mean
        metrics = [self.metric(spectrum, mean_spectrum) for spectrum in spectra]
        # order by metric
        order = np.argsort(metrics)
        spectra_ordered = spectra[order]
        median_spectrum = np.median(spectra_ordered, axis=0)
        # std by pixel
        std = np.std(spectra_ordered, axis=0)
        return mean_spectrum, median_spectrum, std

    def get_cluster_median(self, cluster):
        ''' Get the median spectrum in a cluster '''
        spectra = self.get_spectra_in_cluster(cluster)
        mean_spectrum = np.mean(spectra, axis=0)
        metrics = [self.metric(spectrum, mean_spectrum) for spectrum in spectra]
        order = np.argsort(metrics)
        spectra_ordered = spectra[order]
        median_spectrum = np.median(spectra_ordered, axis=0)
        return median_spectrum
