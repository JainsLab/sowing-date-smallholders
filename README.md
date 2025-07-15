# Mapping Sowing Dates in Smallholder Farming Systems

This repository contains the full code workflow for the research project on estimating crop sowing dates in smallholder agricultural systems using satellite time-series data. The methodology leverages multiple remote sensing products (Sentinel-2, MODIS, and HLS) and applies time-series smoothing and derivative analysis to identify the Start of Season (SOS) as a proxy for the sowing date. The code and workflow are for the paper [**"Mapping grain crop sowing date in smallholder systems using optical imagery"**](https://www.sciencedirect.com/science/article/pii/S2352938525002137) ([Citation.bib](./S2352938525002137.bib)), published in the [*Remote Sensing Applications: Society and Environment*](https://www.sciencedirect.com/journal/remote-sensing-applications-society-and-environment) journal (DOI: 10.1016/j.rsase.2025.101660). 

This repository has a small and uncategorized subset from **Bihar, India**. Some of the field data may be available upon reasonable request, while certain raw data are restricted due to proprietary/privacy concerns.


## 📂 Repository Structure

The project is organized into a clear, reproducible structure:
```
.
.
├── data/
│   └── field_data/         # Contains input survey/field data in different formats (GPKG, geojson)
├── notebooks/
│   ├── 1_data_extraction.ipynb       # Extracts raw time-series data from GEE
│   ├── 2_sowing_date_calculation.ipynb # Cleans, smooths, and calculates phenology. Compares results to survey data and creates final plots
├── output/
│   ├── raw_timeseries/    # Stores raw CSV/Excel data from GEE
│   ├── plots/        # Contains saved plots for each field
│   ├── smoothed_timeseries/        # Stores the smoothed data (in CSVs and .pkl) after daily interpolation and SG/spline.
│   └── sowing_**.csv/          # Final CSV with predicted sowing dates
├── scripts/
│   ├── gee_functions.py          # Helper functions for Google Earth Engine
│   ├── sowing_date_functions.py    # Functions for smoothing and SOS detection
│   └── metric_evaluation_functions.py # Functions for model evaluation (Pontius metrics)
├── requirements.txt          # List of packages used
├── S2352938525002137.bib          # Reference: paper
└── README.md                   # This file
```


## Workflow and How to Run

Follow these steps to set up your environment and run the analysis pipeline from start to finish.

### Step 1: Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JainsLab/sowing-date-smallholders.git
    cd sowing-date-smallholders
    ```

2.  **Create a Python Environment:** It is highly recommended to use a virtual environment (e.g., venv or conda) to manage dependencies.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
    ```

3.  **Install Required Packages:** All necessary packages are listed in [`requirements.txt`](./requirements.txt).
    ```bash
    pip install -r requirements.txt
    ```

4.  **Authenticate Google Earth Engine:** The first time you run the data extraction notebook, you will need to authenticate your computer with GEE.
    ```bash
    ee.Authenticate()
    ```
    Follow the on-screen instructions.

### Step 2: GEE Data Extraction
Run the first notebook to download the raw time-series data for your field polygons.

-   **File:** [`notebooks/1_data_extraction.ipynb`](./notebooks/1_data_extraction.ipynb)
-   **Action:** Open the notebook and be sure to **Authenticate your GEE** at the top. Run all the cells.
-   **Output:** This will generate `.csv` files for each sensor in the [`output/raw_remote_sensing/`](./output/raw_timeseries/) directory. The script is resumable and will skip any fields that have already been processed.

### Step 3: Smoothing and Sowing Date Calculation
This notebook takes the raw data, applies smoothing filters, calculates the sowing date using the derivative method, and saves plots for each field.

-   **File:** [`notebooks/2_sowing_date_calculation.ipynb`](./notebooks/2_sowing_date_calculation.ipynb)
-   **Action:** Run all the cells. This script performs the core analysis. 
-   **Output:**
    -   Smoothed time-series data for each sensor saved as `.csv` files in [`output/raw_remote_sensing/`](./output/raw_timeseries/).
    -   A final `.csv` file with the predicted sowing dates for all models and approaches in [`output/smoothed_timeseries/`](./output/smoothed_timeseries/).
    -   A `.pkl` file containing a dictionary of the final smoothed data.
    -   Individual profile plots for each field saved in [`output/plots/`](./output/plots/).

### Step 4: Metrics and Final Figures
This final step evaluates the model's performance against the ground-truth survey data using the Pontius framework.

-   **File:** [`notebooks/2_sowing_date_calculation.ipynb`](./notebooks/2_sowing_date_calculation.ipynb)
-   **Action:** No action is needed. It will evaluate all the sensors and smoothers.
-   **Output:** A final summary table of performance metrics and a scatter plot comparing predicted vs. survey sowing dates.

---

## License

This project is licensed under the MIT License. See the [`LICENSE`](./LICENSE) file for details.