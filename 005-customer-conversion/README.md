# 🔄 Branching Out: Customer Conversion Prediction with Decision Trees vs. Random Forests
This project applies Decision Tree and Random Forest models to forecast lead-to-sale conversions, generating actionable business recommendations from the best-performing approach.

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
This project implements a supervised machine learning pipeline to predict customer conversion for ExtraaLearn, an early-stage startup. Using Decision Trees and Random Forests, the model captures relationships between customer status and features such as lead age, occupation, and initial interaction platform. The pipeline covers data cleaning, feature engineering, model training, evaluation, and insight extraction, supporting sales prediction, lead analysis, and customer profiling.

## 📊 Dataset

The dataset, sourced from the MIT Applied Data Science Program, is provided in CSV format with 4,612 rows and 15 columns. Each row corresponds to a lead profile and contains details such as age, occupation, first interaction platform with ExtraaLearn, profile completion percentage, number of website visits, total time spent on the website, average pages viewed per visit, last interaction with ExtraaLearn, and media information.

Each record consists of 13 input features describing the lead profile and 1 target variable indicating whether the lead converted to a paid customer.

Detailed feature descriptions are listed in the table below:
| Column                | Description |
|------------------------|-------------|
| **ID**                | Unique ID of the lead |
| **age**               | Age of the lead |
| **current_occupation** | Current occupation of the lead. Values include 'Professional', 'Unemployed', and 'Student' |
| **first_interaction** | How the lead first interacted with ExtraaLearn. Values include 'Website', 'Mobile App' |
| **profile_completed** | Percentage of profile completed on the website/mobile app. Categories: Low (0–50%), Medium (50–75%), High (75–100%) |
| **website_visits**    | Number of times the lead visited the website |
| **time_spent_on_website** | Total time spent on the website |
| **page_views_per_visit** | Average number of pages viewed per website visit |
| **last_activity**     | Last recorded interaction between the lead and ExtraaLearn. Examples: Email Activity (requested program details, received brochure, etc.), Phone Activity (phone/SMS conversation with representative), Website Activity (live chat, profile update, etc.) |
| **print_media_type1** | Flag indicating whether the lead saw ExtraaLearn’s advertisement in a newspaper |
| **print_media_type2** | Flag indicating whether the lead saw ExtraaLearn’s advertisement in a magazine |
| **digital_media**     | Flag indicating whether the lead saw ExtraaLearn’s advertisement on digital platforms |
| **educational_channels** | Flag indicating whether the lead heard about ExtraaLearn through educational channels (online forums, discussion threads, educational websites, etc.) |
| **referral**          | Flag indicating whether the lead heard about ExtraaLearn through a referral |
| **status**            | Conversion status — whether the lead was converted to a paid customer (**Target Variable**) |

## ❓ Problem Statement
High customer conversion—essentially identifying the right leads and approaching them effectively—is always a priority for any company. After all, no one wants to sell a fan to Eskimos; the real goal is to personalize the approach and focus on customers who truly need the product, such as residents in hot climates. Interpreting lead profiles is therefore a crucial business skill, as it enables forecasting which customers are most likely to convert and how to engage less likely customers through tailored strategies. The key to successful interpretation lies in understanding the importance of each lead profile attribute (feature) across a large dataset. The true challenge, however, is determining which model delivers the best performance.

## 🔎 Methodology
The following steps outline the end-to-end process used in this project:

#######################################################################################

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
git clone https://github.com/elescj/005-customer-conversion-lr.git
cd 005-customer-conversion-lr

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the script
python main.py
```
