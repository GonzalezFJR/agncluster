# agncluster: Hierarchical clustering of galaxy datacubes

This repository provides utilities to cluster galaxy integral-field spectroscopy datacubes using hierarchical clustering. The data are 3D cubes where each spatial pixel (spaxel) contains a spectrum. The workflow is:

- Load and prepare the datacube with `modules/cube.py`.
- Compute pairwise spectral distances and perform hierarchical clustering with `modules/hcluster.py`.
- Explore results and examples in the included Jupyter notebook.

---

## Features

- **Datacube handling**: Load FITS cubes with shape `(lambda, y, x)`, build the wavelength array, crop spatial/spectral ranges, and create masks.
- **Normalization options**: Continuum-based normalization, flux normalization, or continuum subtraction prior to clustering.
- **Flexible distance metrics**:
  - `spectra_similarity`: Sum of squared differences (SSD) ignoring NaNs.
  - `spectra_similarity_corr`: SSD after testing shifts in pixels to account for small spectral misalignments (e.g., velocity shifts).
- **Hierarchical clustering**: Uses SciPy linkage (Ward) over pairwise distances, cut into up to `n` clusters with `fcluster`.
- **Convenience helpers**: Retrieve cluster label maps, per-cluster median/mean spectra, integrated spectra, simple pairwise spectrum comparison, and silhouette estimation.

---

## Repository structure

- `modules/cube.py` — Datacube class `cube` to read and prepare FITS spectral cubes.
- `modules/hcluster.py` — Clustering class `hcluster` and distance metrics.
- `example_to_run.ipynb` — End-to-end notebook demonstrating typical usage on a real cube.

---

## Installation

This project is a pure-Python workspace. Install the typical scientific Python stack (Python ≥3.8 recommended):

```bash
pip install numpy scipy astropy scikit-learn matplotlib jupyter
```

You will also need Jupyter to open the example notebook.

---

## Quick start

Below is a minimal example that mirrors the notebook logic.

```python
import numpy as np
from modules.cube import cube
from modules.hcluster import hcluster, spectra_similarity_corr

# 1) Load the FITS datacube (lambda, y, x). ext=1 is common for science data.
cb = cube("/path/to/your_cube.fits", ext=1)

# 2) Optionally restrict spatial/spectral ranges (defaults use the full cube)
#    set_data_limits(xfrom, xto, yfrom, yto, lambda_from, lambda_to)
cb.set_data_limits(0, cb.header['NAXIS1'], 0, cb.header['NAXIS2'], 0, np.inf)

#    You may also define a continuum window in pixel indices (if used for normalization)
cb.set_continuum_limits(710, 730, lambda_idx=True)

#    Apply the crop and prepare internal arrays
cb.cut_data()

# 3) Define a distance metric. Example: shift-aware SSD over ±N pixels
shift_pixels = 6
metric = lambda x, y: spectra_similarity_corr(x, y, N=shift_pixels)

# 4) Build the clustering object. Choose data pre-processing options here
hc = hcluster(
    cb,
    metric=metric,
    n=12,                 # target number of clusters (can be changed later)
    normalize_cont=False, # or True if you want continuum normalization
    normalize_flux=True,  # optional flux normalization
    subtract=False        # or True to subtract continuum instead of normalizing
)

# 5) Compute pairwise distances and then the clustering
hc.compute_distances()      # O(M^2) in number of valid spaxels; can take time for large cubes
hc.compute_clusters(n=8)    # or hc.compute(n=8) as a convenience

# 6) Retrieve cluster labels
labels_1d = hc.get_clusters(matrix=False)   # shape: (valid_spaxels,)
labels_2d = hc.get_clusters(matrix=True)    # shape: (y, x) with NaN at masked pixels

# 7) Example: integrated spectrum of a cluster
spec_int = hc.get_integrated_spectra(cluster=0)
```

Plotting a simple segmentation map (requires matplotlib):

```python
import matplotlib.pyplot as plt
plt.imshow(labels_2d, origin='lower', cmap='tab20')
plt.colorbar(label='Cluster ID')
plt.title('Hierarchical clustering of spectra')
plt.show()
```

---

## Data expectations and masking

- **Input format**: 3D FITS array `(lambda, y, x)`. The default wavelength construction uses the FITS header keywords `CRVAL3`, `CDELT3`, and `CRPIX3`. You can override this with a custom `wavelength_command` when constructing `cube`.
- **Masking**: Pixels with any NaN across the spectral axis or with zeros are masked out for analysis. The mask is available as `cube.cube_mask` after `cut_data()`.
- **Normalization and subtraction**: Control via `hcluster` constructor:
  - `normalize_cont=True` to divide by a continuum level (set with `cube.set_continuum_limits`) before clustering.
  - `normalize_flux=True` to divide each spectrum by its total flux.
  - `subtract=True` to subtract a simple continuum estimate instead of normalizing.

---

## Clustering details

- **Distance computation**: `pdist` is applied to the selected, valid spectra, using the chosen metric.
- **Linkage method**: Ward linkage (`method='ward'`), then `fcluster(..., criterion='maxclust')` to obtain up to `n` clusters.
- **Singleton handling**: The implementation iteratively separates tiny clusters (≤`cluster_singletons`, default 2) to avoid fragmenting the main structure; such spaxels are grouped under a separate label at the end.
- **Performance**: Distance computation scales quadratically with the number of valid spaxels. Consider cropping the region or coarsening data for very large cubes. In practice, medium-size JWST/MIRI MRS cubes can take minutes to tens of minutes.

---

## API highlights

- **From `modules/cube.py`**
  - `cube(filename, wavelength_command=None, ext=1)`
  - `set_data_limits(xfrom, xto, yfrom, yto, lambda_from, lambda_to, lambda_idx=True)`
  - `set_continuum_limits(lambda_from, lambda_to, lambda_idx=True)`
  - `cut_data()`
  - `get_cube(normalize_flux=False, normalize_cont=False, subtract=False)`
  - `get_image(lambda_index, processed=True, log=True, mask=True)`
  - `get_spectrum(y, x, processed=True)`

- **From `modules/hcluster.py`**
  - `hcluster(cube, metric=None, n=12, normalize_cont=True, normalize_flux=False, subtract=False)`
  - `compute_distances()` / `compute(n=None)`
  - `compute_clusters(n=None, cluster_singletons=2, ...)`
  - `get_clusters(matrix=False)`
  - `get_spectra_in_cluster(cluster)` / `get_cluster_median(cluster)` / `get_cluster_size(cluster)` / `get_integrated_spectra(cluster)`
  - `compare(x1, y1, x2, y2)`
  - `spectra_similarity(s1, s2)` and `spectra_similarity_corr(s1, s2, N=10)`

---

## Example notebook

Open `example_to_run.ipynb` in Jupyter to see a full workflow, including loading a real FITS cube, selecting a metric with pixel shifts, computing distances, running clustering, and visualizing the segmentation and representative spectra.

```bash
jupyter notebook example_to_run.ipynb
```

---

## Notes

- FITS header conventions may vary across instruments. If automatic wavelength construction fails, provide an explicit `wavelength_command` (see notebook).
- For reproducibility and speed, consider cropping to a smaller spatial region or spectral range before computing distances.

---

## Acknowledgments

Built for clustering spectral cubes of galaxies, where each pixel hosts a spectrum, enabling data-driven segmentation and spectral characterization across the field.
