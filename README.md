# Manifold Learning for LAMOST Spectra

## Overview

This project aims to explore manifold learning techniques on low-resolution spectral data from the LAMOST (Large Sky Area Multi-Object Fiber Spectroscopic Telescope) survey. The goal is to analyze the intrinsic structure of spectral data for different types of stars (A, F, G).

## Directory Structure

```
Manifold Learning/
├── data/
│   ├── processed/
│   │   ├── AFG_params.csv       # Initial combined parameters (Assumed pre-existing)
│   │   └── sampled/
│   │       ├── type_A_params.csv    # Sampled parameters for Type A
│   │       ├── type_A_obsid.txt     # Sampled obsids for Type A
│   │       ├── type_F_params.csv    # Sampled parameters for Type F
│   │       ├── type_F_obsid.txt     # Sampled obsids for Type F
│   │       ├── type_G_params.csv    # Sampled parameters for Type G
│   │       ├── type_G_obsid.txt     # Sampled obsids for Type G
│   │       ├── AFG_merged_params.csv # Combined sampled parameters
│   │       └── AFG_merged_obsid.txt  # Combined sampled obsids
│   └── spectra/
│       ├── type_A/                # Downloaded FITS files for Type A
│       ├── type_F/                # Downloaded FITS files for Type F
│       ├── type_G/                # Downloaded FITS files for Type G
│       └── type_*_failed_obsids.txt # List of failed downloads
├── src/
│   ├── data_sample.py           # Script to sample data by star type
│   └── download_spectra.py      # Script to download spectra using pylamost
├── pylamost.py                  # LAMOST data access library (custom)
└── README.md                    # This file
```

## Setup and Installation

1.  **Python Environment:**
    *   It is recommended to use a virtual environment (e.g., conda or venv).
    *   This project has been tested with Python 3.10.
    ```bash
    # Example using conda
    conda create -n lamost_manifold python=3.10
    conda activate lamost_manifold
    ```

2.  **Install Dependencies:**
    *   Install required Python packages:
        ```bash
        pip install pandas numpy astropy scipy matplotlib requests
        ```

3.  **`pylamost` Library:**
    *   The custom `pylamost.py` library is used for accessing LAMOST data.
    *   Currently, it needs to be manually placed in the Python environment's `site-packages` directory. Find your site-packages path:
        ```bash
        python -c "import site; print(site.getsitepackages())"
        ```
    *   Copy the file (replace `<site-packages-path>` with the actual path found above):
        ```bash
        cp pylamost.py <site-packages-path>/
        ```
    *   *Note: A better approach would be to package `pylamost` properly or include it directly in the `src` directory and adjust imports.* 

## Workflow / Usage

1.  **Prepare Initial Data:**
    *   Ensure the initial combined parameter file `data/processed/AFG_params.csv` exists. This file should contain columns like `obsid`, `subclass`, `teff`, `logg`, `feh`, `snrg`.

2.  **Sample Data:**
    *   Run the sampling script to select a subset of data for each star type (A, F, G) and generate parameter/obsid files in `data/processed/sampled/`.
    ```bash
    python src/data_sample.py
    ```

3.  **Download Spectra:**
    *   Run the download script. It uses the obsid lists generated in the previous step to download the corresponding low-resolution FITS spectra from the LAMOST archive (currently set to use the DR10 API).
    *   **Important:** The script `src/download_spectra.py` currently has a **hardcoded LAMOST token**. You might need to update it if the token expires or changes.
    ```bash
    python src/download_spectra.py 
    ```
    *   Downloads will be saved to the `data/spectra/` directory, organized by star type.
    *   Failed downloads will be logged in `type_*_failed_obsids.txt` files within the `data/spectra/` directory.

4.  **Manifold Learning Analysis:**
    *   (Future Step) Use the downloaded FITS files and sampled parameters for manifold learning analysis.

## Data

*   **Source:** LAMOST Low-Resolution Spectroscopic Survey (DR10 used for download API).
*   **Star Types:** A, F, G type stars.
*   **Parameters:** `teff` (Effective Temperature), `logg` (Surface Gravity), `feh` (Metallicity), `snrg` (Signal-to-Noise Ratio in g-band).
*   **Spectra:** Low-resolution FITS files.

## Notes

*   The LAMOST data download API can sometimes be unstable or experience downtime, leading to `HTTP 500` errors during download. Re-running the download script later or using the `failed_obsids.txt` files for targeted re-downloads might be necessary.
*   The LAMOST token is currently hardcoded in `src/download_spectra.py`.
