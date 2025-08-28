# Exploratory Data Analysis (EDA) on Dataset "House Prices: Advanced Regression Techniques"

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-3696?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)

## 📊 Project Overview

This project performs an exploratory data analysis (EDA) and data preprocessing on the Ames Housing dataset from a Kaggle competition. The primary goal is to understand the relationships between various housing features and the target variable, `SalePrice`, and to prepare the data for a predictive modeling task.

## 🏗️ Key Steps and Methodology

The analysis follows a structured data science pipeline:

### 1. Import Libraries & Load Data
- **Libraries Used:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `sklearn`
- **Data Source:** The Kaggle dataset `train.csv` is loaded into a DataFrame `df_train`.

### 2. Initial Data Exploration
- **Target Variable Analysis:** The `SalePrice` column is examined using descriptive statistics (`describe()`) and visualized with a histogram (`distplot`).
- **Findings:** The initial sale price distribution is right-skewed (Skewness: 1.88) with heavy tails (Kurtosis: 6.54), indicating a deviation from normality.

### 3. Bivariate Analysis
Relationships between potential predictors and `SalePrice` are explored visually:
- **Scatter Plots:** Used for continuous variables (`GrLivArea`, `TotalBsmtSF`). Revealed positive correlations and identified two severe outliers with extremely large `GrLivArea` but low `SalePrice`.
- **Box Plots:** Used for categorical/ordinal variables (`OverallQual`, `YearBuilt`). Clearly showed that `OverallQual` is a strong predictor of price.

### 4. Correlation Analysis
- A correlation matrix heatmap was generated for all numerical variables.
- The top 10 features most correlated with `SalePrice` were isolated and visualized in a focused heatmap. The strongest correlates are:
    1. `OverallQual`
    2. `GrLivArea`
    3. `GarageCars`
    4. `GarageArea`
    5. `TotalBsmtSF`

### 5. Missing Data Handling
- Missing values were counted and sorted. Columns with more than 1 missing value (e.g., `PoolQC`, `MiscFeature`, `Alley`) were **dropped entirely**.
- The one remaining missing value in the `Electrical` column was handled by **dropping that single row**.
- The final cleaned dataset has **1,459 rows and 63 columns**.

### 6. Outlier Treatment
- Based on the scatter plot, two obvious outliers (IDs 1299 and 524) were identified and removed from the dataset to prevent them from skewing the model.

### 7. Data Transformation for Normality
To make the data more suitable for linear models, log transformations were applied:
- **`SalePrice`:** Transformed using `np.log` to correct for strong right-skewness.
- **`GrLivArea`:** Also log-transformed to normalize its distribution.
- **`TotalBsmtSF`:** A special approach was used for this variable, which contains zeros (houses with no basement). A new boolean feature `HasBsmt` was created, and then the log transformation was applied only to records where `TotalBsmtSF` > 0.

### 8. Encoding Categorical Variables
- All remaining categorical variables were converted into a machine-readable format using **one-hot encoding** (`pd.get_dummies`).

## 📈 Results and Conclusions

- The distribution of the target variable (`SalePrice`) and major continuous features (`GrLivArea`, `TotalBsmtSF`) was successfully normalized using log transformations.
- The strongest predictors of house price in this dataset are **Overall Quality**, **Above-Ground Living Area**, and **Garage Size**.
- The dataset has been cleaned of missing values and outliers.
- The data is now preprocessed and ready for further feature selection and modeling using algorithms like Linear Regression, Ridge/Lasso, or Random Forests.

## 🚀 Next Steps

1.  **Feature Engineering:** Create new features from existing ones (e.g., age of the house, total bathroom count).
2.  **Feature Selection:** Use techniques like Recursive Feature Elimination (RFE) to select the most important variables.
3.  **Modeling:** Apply various regression models and tune their hyperparameters.
4.  **Validation:** Use cross-validation to ensure the model's robustness and test it on the holdout `test.csv` dataset.

## 📁 Files

- `train.csv`: The original training data.
- The provided Jupyter Notebook/Python script contains all the code for the analysis.

## 🛠️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/your-username/house-prices-eda.git

# Navigate to the directory
cd house-prices-eda

# Install required packages
pip install -r requirements.txt

# Run the Jupyter notebook
jupyter notebook house_prices_eda.ipynb
