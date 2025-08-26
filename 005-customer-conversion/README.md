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

## 📊 Model Evaluation

The models were evaluated on both **training** and **test** datasets using standard classification metrics.  

- **Precision (1)**: When the model predicts a conversion, how often it is correct.  
- **Recall (1)**: Out of all actual conversions, how many were correctly identified.  
- **F1-Score (1)**: Balance between precision and recall.  
- **Accuracy**: Overall correct predictions (can be misleading if classes are imbalanced).  

---

### 🌲 Decision Tree – Training Data
| Class                        | Precision | Recall | F1-Score | Support |
|------------------------------|-----------|--------|----------|---------|
| 0 (converted to sale)        | 0.94      | 0.77   | 0.85     | 2273    |
| 1 (not converted to sale)    | 0.62      | 0.88   | 0.73     | 955     |
| **Accuracy**                 |           |        | **0.80** | 3228    |
| **Macro Avg**                | 0.78      | 0.83   | 0.79     | 3228    |
| **Weighted Avg**             | 0.84      | 0.80   | 0.81     | 3228    |

---

### 🌲 Decision Tree – Test Data
| Class                        | Precision | Recall | F1-Score | Support |
|------------------------------|-----------|--------|----------|---------|
| 0 (converted to sale)        | 0.93      | 0.77   | 0.84     | 962     |
| 1 (not converted to sale)    | 0.62      | 0.86   | 0.72     | 422     |
| **Accuracy**                 |           |        | **0.80** | 1384    |
| **Macro Avg**                | 0.77      | 0.82   | 0.78     | 1384    |
| **Weighted Avg**             | 0.83      | 0.80   | 0.80     | 1384    |

---

### 🏕️ Random Forest – Training Data
| Class                        | Precision | Recall | F1-Score | Support |
|------------------------------|-----------|--------|----------|---------|
| 0 (converted to sale)        | 0.94      | 0.83   | 0.88     | 2273    |
| 1 (not converted to sale)    | 0.68      | 0.87   | 0.76     | 955     |
| **Accuracy**                 |           |        | **0.84** | 3228    |
| **Macro Avg**                | 0.81      | 0.85   | 0.82     | 3228    |
| **Weighted Avg**             | 0.86      | 0.84   | 0.84     | 3228    |

---

### 🏕️ Random Forest – Test Data
| Class                        | Precision | Recall | F1-Score | Support |
|------------------------------|-----------|--------|----------|---------|
| 0 (converted to sale)        | 0.93      | 0.83   | 0.87     | 962     |
| 1 (not converted to sale)    | 0.68      | 0.85   | 0.76     | 422     |
| **Accuracy**                 |           |        | **0.83** | 1384    |
| **Macro Avg**                | 0.81      | 0.84   | 0.82     | 1384    |
| **Weighted Avg**             | 0.85      | 0.83   | 0.84     | 1384    |

---

### 📌 Model Performance Summary (Test Data)

| Model           | Accuracy | Precision (1) | Recall (1) | F1 (1) |
|-----------------|----------|---------------|------------|--------|
| Decision Tree   | 0.80     | 0.62          | 0.86       | 0.72   |
| Random Forest   | 0.83     | 0.68          | 0.85       | 0.76   |

---

#######################################################################################

### 💡 Insights

- The **Decision Tree** achieves reasonable accuracy but may overfit compared to Random Forest.  
- The **Random Forest** generally shows stronger performance, with higher F1-scores and better generalization.  
- Recall for **class 1 (converted customers)** is especially important, as capturing more potential conversions directly impacts business outcomes.  

---

✅ Just replace the `...` placeholders with your sklearn report numbers.  
✅ If you computed confusion matrices or feature importance plots, you can add them after the tables for visuals.  

---

Do you want me to also create a **version with collapsible sections** (so the big metric tables can be folded in Markdown for readability)?

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
