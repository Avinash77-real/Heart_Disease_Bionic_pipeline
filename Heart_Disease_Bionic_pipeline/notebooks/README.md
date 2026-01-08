# Experimental Data Analysis & Research Notebooks

This folder contains the interactive research and visualization components of the Heart Disease Diagnostic Pipeline. These notebooks document the "discovery phase," where raw medical data is analyzed to inform the design of the cloud-based controller.

## 📓 Notebook Directory

### 1. [EDA & Statistical Discovery](./01_exploratory_data_analysis.ipynb)
**Goal:** To perform Exploratory Data Analysis (EDA) and verify data integrity for clinical use.
- **Statistical Profiling:** Analysis of mean, variance, and distribution of patient vitals (Age, Trestbps, Chol).
- **Feature Correlation:** Identifying which biometric markers (e.g., `thalach`, `cp`) have the highest predictive power for heart disease.
- **Prerequisite Check:** Verified zero null values and addressed class balance (165 Diseased vs 138 Healthy).

### 2. [Model Development & Benchmarking](./02_model_development.ipynb)
**Goal:** To experiment with the Machine Learning pipeline before modularization.
- **Pipeline Testing:** Implementation of `StandardScaler` and `LogisticRegression`.
- **Performance Evaluation:** Generation of Confusion Matrices and Classification Reports to ensure high **Recall** (Sensitivity), which is critical in healthcare applications.

## 📊 Key Visualizations

### Feature Correlation Matrix
This heatmap was used to identify the relationship between input features. In a bionic context, this helps in **sensor selection**—determining which ambient sensors provide the most useful data for the controller.



### The ML Pipeline Architecture
This diagram outlines the flow from raw data ingestion to the final predictive output, simulating the communication between a bionic limb and a cloud server.



## 🛠️ Requirements for Notebooks
To run these notebooks locally, ensure you have the following installed:
- `jupyter`
- `seaborn`
- `matplotlib`
- `pandas`