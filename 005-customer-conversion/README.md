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

1. **Initial Data Treatment**
   - **Data Overview**: Examined the dataset to understand variable types, identify duplicates, and assess missing values.
   - **Initial Data Treatment**: Cleaned and structured the dataset to ensure readiness for exploratory data analysis (EDA).

3. **Exploratory Data Analysis (EDA)**
   - **Univariate Analysis**: Analyzed the distribution of numerical and categorical variables using histograms and descriptive statistics.
   - **Bivariate Analysis**: Investigated relationships between independent variables and the target by:
      - Using correlation heatmaps for numerical variables.
      - Comparing numerical variable distributions across target classes with histograms and boxplots.
      - Assessing conversion rates of categorical variables with stacked bar plots.

4. **Modelling**
   - Trained **Decision Tree** and **Random Forest** models separately to forecast customer conversion.
   - Tuned **hyperparameters** to improve predictive performance and mitigate overfitting.
   - Evaluated model performance using relevant metrics and visualized the tuned results.

#######################################################################################

## 📈 Results

### 💯 Model Performance Metrics

| Model         | Class             | Precision | Recall | F1-Score | Support |
|---------------|-------------------|-----------|--------|----------|---------|
| Decision Tree | 0 (Not Converted) | 0.94      | 0.77   | 0.85     | 2273    |
|               | 1 (Converted)     | 0.62      | 0.88   | 0.73     |  955    |
| Random Forest        | 0.9799    | 0.8726   | 1.5830      | 3.9787     |

| Class |  |  |  | Support |
|-------|-----------|--------|----------|---------|



**Overall Performance**
DTree
| Metric        | Score |
|---------------|-------|
| Accuracy      | 0.80  |
| Macro Avg F1  | 0.79  |
| Weighted Avg F1 | 0.81 |

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
