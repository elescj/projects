# 🔄 Branching Out: Customer Conversion Prediction with Decision Trees vs. Random Forests
------------------------------------------
This project applies Linear Regression, Ridge Regression, Decision Tree, Random Forest, and KNN Regression to forecast used car prices in India based on key parameters, providing actionable business recommendations using the best-performing regression model.

## 📂 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Problem Statement](#-problem-statement)
- [Methodology](#-methodology)
- [Results](#-results)
- [Insights & Recommendations](#-insights--recommendations)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)

## 🧠 Overview
This project implements a supervised machine learning pipeline to predict used car prices in India, leveraging multiple regression methods to model the relationship between price and features such as car name, location, and manufacturing year. The pipeline includes data cleaning, feature engineering, model training, evaluation, and insights extraction to support pricing decisions in the used car market.

## 📊 Dataset
This dataset, originally provided in the Applied Data Science Program by MIT, is a CSV file with 7,253 rows and 14 columns, where each row represents a used car sold in India, including details such as car model, location, year of manufacture, mileage, engine capacity, and selling price.

Each record includes **13 input features** describing property and neighborhood characteristics, and one **target variable**: the median value of owner-occupied homes (in $1000s).

Detailed feature descriptions are listed in the table below:
| Column | Description |
|--------|-------------|
| **S.No.** | Serial Number |
| **Name** | Car name, including brand and model |
| **Location** | City where the car is available |
| **Year** | Manufacturing year |
| **Kilometers_Driven** | Total kilometers driven by previous owner(s) |
| **Fuel_Type** | Fuel type (Petrol, Diesel, Electric, CNG, LPG) |
| **Transmission** | Transmission type (Automatic/Manual) |
| **Owner_Type** | Type of ownership (First, Second, etc.) |
| **Mileage** | Mileage in kmpl or km/kg |
| **Engine** | Engine displacement in CC |
| **Power** | Engine power in BHP |
| **Seats** | Number of seats |
| **New_Price** | Price of a new car of the same model (in INR 100,000) |
| **Price** | Price of the used car (in INR 100,000) (**Target Variable**) |

## ❓ Problem Statement
Automobiles are essential in modern society, enabling daily commutes, logistics, and long-distance travel. As the population grows and urbanization accelerates, the demand for personal vehicles can no longer be met solely by the new car market. Used cars, offering practical value at lower costs, have become a popular choice for many buyers.

However, pricing used cars is challenging. Even well-maintained cars depreciate over time due to factors such as mileage, brand, model, year, and market demand. Determining the right price for each used car typically requires thorough inspection and market analysis, which is time-consuming and costly. Inaccurate pricing risks eroding customer trust and impacting sales.

Thus, there is a critical need for an **accurate, scalable, and low-maintenance pricing mechanism** to support fair pricing in the used car market, benefiting both sellers and buyers.

## 🔎 Methodology
The following steps outline the end-to-end process used in this project:

1. **Initial Data Treatment**
   - **Data Overview**: Reviewed to understand data types, check for duplicates, and assess missing values.
   - Clarified **action items for EDA** based on initial observations.
   
3. **Exploratory Data Analysis (EDA)**
   - **Univariate Analysis**: Examined the distribution of numerical and categorical variables using histograms and descriptive statistics.
   - **Bivariate Analysis**: Explored relationships between independent variables and the target (`Price`) using scatter plots and correlation heatmaps to identify potential multicollinearity and feature relevance.
   - **Feature Engineering**: Transformed vague or inconsistent variables into more meaningful, usable forms for modeling (e.g., extracting car brand, creating `car_age`).
   - Continued **data treatment**:
     - Dropped purely cardinal or irrelevant variables.
     - Imputed missing values using median, mean, or model-based methods to ensure a clean dataset for modeling.

4. **Model Building**
   - Performed **Linear Regression, Ridge Regression, Decision Tree, Random Forest, and KNN Regression** separately to forecast used car prices.
   - Tuned **hyperparameters** for each model to enhance predictive performance and reduce overfitting.
   - Compared **R² Scores and RMSE** across all models to evaluate and select the best-performing method for price prediction.

## 📈 Results

### 💯 Model Performance Metrics

| Model               | Train R²  | Test R²  | Train RMSE | Test RMSE |
|----------------------|-----------|----------|-------------|------------|
| Linear Regression    | 0.9835    | 0.8197   | 1.4346      | 4.7319     |
| Ridge                | 0.9666    | 0.9098   | 2.0432      | 3.3478     |
| Decision Tree        | 0.9597    | 0.8489   | 2.2437      | 4.3319     |
| Random Forest        | 0.9799    | 0.8726   | 1.5830      | 3.9787     |
| KNN                  | 0.9999    | 0.8316   | 0.0203      | 4.5729     |

**Ridge Regression achieves the highest Test R² (`0.910`) and the lowest Test RMSE (`3.35`)**, indicating **best generalization** and **predictive accuracy** on unseen data.

---

### 🔢 Model Coefficient

| Parameter | Default | Description |
|---|---|---|
| **`alpha`** | `1.0` | Regularization strength; higher = more regularization, helping reduce overfitting. |
| **`fit_intercept`** | `True` | Whether to calculate the intercept (`b0`). |
| **`normalize`** | *deprecated* | Previously normalized features automatically; now use `StandardScaler` explicitly. |
| **`solver`** | `'auto'` | Algorithm for optimization. Options include `'auto'`, `'svd'`, `'cholesky'`, `'lsqr'`, `'sparse_cg'`, `'sag'`, `'saga'`, `'lbfgs'`. |
| **`max_iter`** | `None` | Maximum iterations for solvers that support it (`'sag'`, `'saga'`). |
| **`tol`** | `1e-3` | Precision tolerance for convergence. |
| **`random_state`** | `None` | Ensures reproducibility for stochastic solvers (`'sag'`, `'saga'`). |
| **`copy_X`** | `True` | Whether to copy `X` or overwrite it. |
| **`positive`** | `False` | Forces coefficients to remain positive if set to `True`. |

## 💡 Insights & Recommendations

- Builds **trust** with fair, data-backed pricing.  
- **Saves time** by reducing manual pricing work.  
- **Boosts profits** with data-aligned buying and selling.  
- Enables **targeted marketing** using high-value features.

<a id="technologies-used"></a>
## ⚙️ Technologies Used
- **Python** – General purpose programming
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computations
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Model training and evaluation
- **Statsmodels** – Statistical modeling (OLS regression, VIF)
- **SciPy** – Statistical tests and probability distributions
- **Warnings** – For suppressing non-critical output

<a id="how-to-run"></a>
## ▶️ How to Run
```bash
# Clone the repository
git clone https://github.com/elescj/003-boston-housing-lr.git
cd 003-boston-housing-lr

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the script
python main.py
```
