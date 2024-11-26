import numpy as np
from scipy.spatial.distance import pdist, squareform
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

    def get_data(self, masked=True):
        mask1d = self.cube.cube_mask.reshape(-1)
        return  self.data[mask1d] if masked else self.data

    def reshape(self, data1d, unmask=True):
        if unmask:
            mask1d = self.cube.cube_mask.reshape(-1)
            #data2d = np.zeros(mask1d.shape)
            # nans array
            data2d = np.full(mask1d.shape, np.nan)
            data2d[mask1d] = data1d
        else:
            data2d = data1d
        return data2d.reshape(self.shape[1:])

    def compute_distances(self):
        self.distances = pdist(self.get_data(), metric=self.metric)

    def compute_clusters(self, n=None, cluster_singletons=2, distances=None, other_indices=None):

        if distances is None:
            distances = self.distances
        if n is None:
            n = self.n  # Assuming self.n is set elsewhere
        if other_indices is None:
            other_indices = np.array([], dtype=int)

        # Total number of data points
        N_total = int((1 + np.sqrt(1 + 8 * len(self.distances))) / 2)

        # Determine the indices of the data points currently being clustered
        if len(other_indices) == 0:
            current_indices = np.arange(N_total)
        else:
            current_indices = np.setdiff1d(np.arange(N_total), other_indices)

        # If only one data point remains, assign it and return
        if len(current_indices) <= 1:
            cluster_labels_full = np.full(N_total, -1)
            if len(current_indices) == 1:
                cluster_labels_full[current_indices] = 0
            # Assign singleton indices to a new cluster label
            singleton_cluster_label = 1
            cluster_labels_full[other_indices] = singleton_cluster_label
            # Map labels to consecutive integers starting from 0
            unique_labels = np.unique(cluster_labels_full)
            self.cluster_labels = np.searchsorted(unique_labels, cluster_labels_full)
            self.linkage = None  # No linkage possible with one data point
            return

        # If distances do not correspond to current_indices, recompute distances
        num_points = len(current_indices)
        expected_condensed_size = num_points * (num_points - 1) // 2
        if len(distances) != expected_condensed_size:
            # Recompute distances for current_indices
            full_distance_matrix = squareform(self.distances)
            distance_matrix_current = full_distance_matrix[np.ix_(current_indices, current_indices)]
            distances = squareform(distance_matrix_current)

        # Compute the clustering
        linkage_matrix = linkage(distances, method='ward')
        cluster_labels = fcluster(linkage_matrix, n, criterion='maxclust')

        # Identify singleton clusters
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        singleton_labels = unique_labels[counts <= cluster_singletons]

        # Get indices of singleton and non-singleton clusters
        singleton_mask = np.isin(cluster_labels, singleton_labels)
        singleton_current_indices = current_indices[singleton_mask]
        non_singleton_current_indices = current_indices[~singleton_mask]

        # Update 'other_indices' with new singleton indices
        other_indices = np.concatenate([other_indices, singleton_current_indices])

        # Check if new singletons were found
        if len(singleton_labels) > 0 and len(non_singleton_current_indices) > 0:
            # Reduce 'n' to avoid requesting more clusters than data points
            n_new = min(n, len(non_singleton_current_indices))
            # Remove singleton data and recurse
            self.compute_clusters(n=n_new, cluster_singletons=cluster_singletons,
                                distances=None, other_indices=other_indices)
        else:
            # Assemble the final cluster labels
            cluster_labels_full = np.full(N_total, -1)
            if len(non_singleton_current_indices) > 0:
                # Map non-singleton cluster labels back to original indices
                cluster_labels_full[non_singleton_current_indices] = cluster_labels[~singleton_mask]
                max_label = cluster_labels[~singleton_mask].max()
            else:
                max_label = 0
            # Assign singleton indices to a new cluster label
            singleton_cluster_label = max_label + 1
            cluster_labels_full[other_indices.astype(int)] = singleton_cluster_label
            # Map labels to consecutive integers starting from 0
            unique_labels = np.unique(cluster_labels_full)
            self.cluster_labels = np.searchsorted(unique_labels, cluster_labels_full)
            # Update the linkage matrix
            self.linkage = linkage_matrix if len(non_singleton_current_indices) > 1 else None
            return


        '''
        print(f'Shape of distances: {self.distances.shape}')

        self.linkage = linkage(distances, method='ward')
        self.cluster_labels = fcluster(self.linkage, self.n, criterion='maxclust')

        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        singleton_labels = unique_labels[counts <= cluster_singletons]

        if len(singleton_labels) > 0:
            # Assign a new cluster label for anomalies
            anomaly_label = -1
            for label in singleton_labels:
                self.cluster_labels[self.cluster_labels == label] = anomaly_label
            # rename cluster labels to be consecutive -- remove labels in singleton_labels
            unique_labels = np.unique(self.cluster_labels)
            # sort
            unique_labels = np.sort(unique_labels)
            for i, label in enumerate(unique_labels):
                self.cluster_labels[self.cluster_labels == label] = i

        print(f'Shape of cluster labels: {self.cluster_labels.shape}')
        '''

    def compute(self, n=None):
        self.compute_distances()
        self.compute_clusters(n)

    def get_clusters(self, matrix=False):
        return self.cluster_labels if not matrix else self.reshape(self.cluster_labels)

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
