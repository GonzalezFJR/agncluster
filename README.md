# agncluster: Hierarchical clustering of galaxy datacubes

This repository provides utilities to cluster galaxy integral-field spectroscopy datacubes using hierarchical clustering. The method is described in https://arxiv.org/pdf/2509.14019.

The data are 3D cubes where each spatial pixel (spaxel) contains a spectrum. The workflow is:

- Load and prepare the datacube with `modules/cube.py`.
- Compute pairwise spectral distances and perform hierarchical clustering with `modules/hcluster.py`.

You can explore results and examples using the included Jupyter notebook.


## Repository structure and Installation

This project is a pure-Python workspace. Install the typical scientific Python stack (Python ≥3.8 recommended):

```bash
pip install numpy scipy astropy scikit-learn matplotlib jupyter
```

You will also need Jupyter to open the example notebook.

The repository contains the following files:

- `modules/cube.py` — Datacube class `cube` to read and prepare FITS spectral cubes.
- `modules/hcluster.py` — Clustering class `hcluster` and distance metrics.
- `example_to_run.ipynb` — End-to-end notebook demonstrating typical usage on a real cube.


## Features

- **Datacube handling**: Load FITS cubes with shape `(lambda, y, x)`, build the wavelength array, crop spatial/spectral ranges, and create masks.
- **Normalization options**: Continuum-based normalization, flux normalization, or continuum subtraction prior to clustering.
- **Flexible distance metrics**:
  - `spectra_similarity`: Sum of squared differences (SSD) ignoring NaNs.
  - `spectra_similarity_corr`: SSD after testing shifts in pixels to account for small spectral misalignments (e.g., velocity shifts).
- **Hierarchical clustering**: Uses SciPy linkage (Ward) over pairwise distances, cut into up to `n` clusters with `fcluster`.
- **Convenience helpers**: Retrieve cluster label maps, per-cluster median/mean spectra, integrated spectra, simple pairwise spectrum comparison, and silhouette estimation.


### Data expectations and masking

- **Input format**: 3D FITS array `(lambda, y, x)`. The default wavelength construction uses the FITS header keywords `CRVAL3`, `CDELT3`, and `CRPIX3`. You can override this with a custom `wavelength_command` when constructing `cube`.
- **Masking**: Pixels with any NaN across the spectral axis or with zeros are masked out for analysis. The mask is available as `cube.cube_mask` after `cut_data()`.
- **Normalization and subtraction**: Control via `hcluster` constructor:
  - `normalize_cont=True` to divide by a continuum level (set with `cube.set_continuum_limits`) before clustering.
  - `normalize_flux=True` to divide each spectrum by its total flux.
  - `subtract=True` to subtract a simple continuum estimate instead of normalizing.

### Clustering details

- **Distance computation**: `pdist` is applied to the selected, valid spectra, using the chosen metric.
- **Linkage method**: Ward linkage (`method='ward'`), then `fcluster(..., criterion='maxclust')` to obtain up to `n` clusters.
- **Singleton handling**: The implementation iteratively separates tiny clusters (≤`cluster_singletons`, default 2) to avoid fragmenting the main structure; such spaxels are grouped under a separate label at the end.
- **Performance**: Distance computation scales quadratically with the number of valid spaxels. Consider cropping the region or coarsening data for very large cubes. In practice, medium-size JWST/MIRI MRS cubes can take minutes to tens of minutes.
