# 🏦 Predicting Loan Default: A Comparative Study of ML Models
This project applies Logistic Regression, Decision Tree, and Random Forest models to forecast loan default, generating actionable business recommendations from the best-performing approach.
![Graphical Summary](attachments/loan.png)

## 📂 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Problem Statement](#-problem-statement)
- [Methodology](#-methodology)
- [Results](#-results)
- [Insights & Recommendations](#-insights--recommendations)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)

## 👓 Overview
This project implements a supervised machine learning pipeline to predict load default for a retail bank. Using Logistic Regression, Decision Trees, and Random Forests, the model captures relationships between loan default and features such as amount of loan approved, amount due on the existing mortgage, current value of the property, and so on. The pipeline covers data cleaning and treatment, model training, evaluation, and insight extraction, supporting loan default prediction.

## 📊 Dataset

The dataset, sourced from the MIT Applied Data Science Program, is provided in CSV format with 5,960 rows and 13 columns. Each row corresponds to an applicant profile and contains details such as amount of loan approved, amount due on the existing mortgage, current value of the property, and so on.

Each record consists of 12 input features describing the applicant profile and 1 target variable indicating whether the client is defaulted on loan.

Detailed feature descriptions are listed in the table below:
| Column      | Description                                                                                                                                                                   |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **BAD**     | Loan status: 1 = Client defaulted on loan, 0 = Loan repaid                                                                                                                    |
| **LOAN**    | Amount of loan approved                                                                                                                                                       |
| **MORTDUE** | Amount due on the existing mortgage                                                                                                                                           |
| **VALUE**   | Current value of the property                                                                                                                                                 |
| **REASON**  | Reason for the loan request <br> - `HomeImp` = Home improvement <br> - `DebtCon` = Debt consolidation (taking out a new loan to pay off other liabilities and consumer debts) |
| **JOB**     | Type of job the loan applicant has (e.g., manager, self-employed, etc.)                                                                                                       |
| **YOJ**     | Years at present job                                                                                                                                                          |
| **DEROG**   | Number of major derogatory reports (serious delinquency or late payments)                                                                                                     |
| **DELINQ**  | Number of delinquent credit lines <br> A line of credit becomes delinquent when a borrower does not make the minimum required payments 30–60 days past due                    |
| **CLAGE**   | Age of the oldest credit line in months                                                                                                                                       |
| **NINQ**    | Number of recent credit inquiries                                                                                                                                             |
| **CLNO**    | Number of existing credit lines                                                                                                                                               |
| **DEBTINC** | Debt-to-income ratio (All monthly debt payments ÷ gross monthly income). This measures a borrower’s ability to manage monthly payments to repay the loan                      |


## ❓ Problem Statement
Predicting loan defaults is important for helping lenders make smarter decisions and reduce financial risk. It’s a challenging problem because people’s financial situations and repayment behaviors are complex, influenced by factors like credit history, debt, employment, and the purpose of the loan. In this project, we use historical loan data to build a machine learning model that predicts whether an applicant will default or repay, providing practical insights to guide lending decisions.

## 💻 Methodology
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
   - **Data Treatment**: Imputed the missing value and treated outliers.

4. **Modelling**
   - Trained **Logistic Regression**, **Decision Tree**, and **Random Forest** models separately to forecast customer conversion.
   - Tuned **hyperparameters** to improve predictive performance and mitigate overfitting.
   - Evaluated model performance using relevant metrics and visualized the tuned results.

## 📝 Results

The models were evaluated on both **training** and **test** datasets using standard classification metrics.  

- **Precision (1)**: When the model predicts a conversion, how often it is correct.  
- **Recall (1)**: Out of all actual conversions, how many were correctly identified.  
- **F1-Score (1)**: Balance between precision and recall.  
- **Accuracy**: Overall correct predictions (can be misleading if classes are imbalanced).  

| Model                 | Dataset  | Class | Precision | Recall | F1-Score | Support | Accuracy | Macro Avg (F1) | Weighted Avg (F1) |
| --------------------- | -------- | ----- | --------- | ------ | -------- | ------- | -------- | -------------- | ----------------- |
| Logistic Regression   | Training | 0     | 0.85      | 0.96   | 0.90     | 3355    | 0.83     | 0.66           | 0.81              |
| Logistic Regression   | Training | 1     | 0.66      | 0.30   | 0.42     | 817     |          |                |                   |
| Logistic Regression   | Test     | 0     | 0.83      | 0.97   | 0.90     | 1416    | 0.82     | 0.64           | 0.79              |
| Logistic Regression   | Test     | 1     | 0.69      | 0.27   | 0.39     | 372     |          |                |                   |
| Decision Tree (tuned) | Training | 0     | 0.94      | 0.90   | 0.92     | 3355    | 0.87     | 0.81           | 0.88              |
| Decision Tree (tuned) | Training | 1     | 0.65      | 0.76   | 0.70     | 817     |          |                |                   |
| Decision Tree (tuned) | Test     | 0     | 0.93      | 0.90   | 0.91     | 1416    | 0.87     | 0.80           | 0.87              |
| Decision Tree (tuned) | Test     | 1     | 0.66      | 0.73   | 0.69     | 372     |          |                |                   |
| Random Forest (tuned) | Training | 0     | 0.94      | 0.89   | 0.92     | 3355    | 0.87     | 0.81           | 0.88              |
| Random Forest (tuned) | Training | 1     | 0.63      | 0.77   | 0.70     | 817     |          |                |                   |
| Random Forest (tuned) | Test     | 0     | 0.92      | 0.90   | 0.91     | 1416    | 0.86     | 0.80           | 0.86              |
| Random Forest (tuned) | Test     | 1     | 0.65      | 0.72   | 0.68     | 372     |          |                |                   |

## 💡 Insights & Recommendations

### 🔎 Insights
- **Random Forest outperforms Decision Tree** across all metrics on the test set, confirming that the ensemble approach reduces overfitting and improves generalization.  
- Both models achieve **strong recall for conversions (class 1)** — 0.86 (Decision Tree) and 0.85 (Random Forest). This means they capture most actual customers, which is valuable for minimizing missed opportunities.  
- **Precision for conversions is lower** (0.62 DT, 0.68 RF), indicating that a non-trivial number of non-converting leads are still predicted as converters. This could result in sales effort being spent on weaker leads. 
- **Accuracy** is higher for Random Forest (0.83 vs. 0.80), showing a better balance between identifying converters and non-converters.  
- **Business takeaway**: If the goal is to **maximize conversion opportunities**, Random Forest is preferred, as it maintains high recall while offering better precision than Decision Tree.

### ✅ Recommendations
- **Deploy Random Forest as the primary model** for predicting customer conversions, as it provides higher accuracy, better precision, and strong recall compared to Decision Tree.  
- **Prioritize leads flagged as high probability converters** by the model, while acknowledging that some false positives may occur.  
- **Consider further feature engineering or additional data sources** to improve precision, reducing unnecessary sales effort on unlikely conversions.  
- **Regularly retrain the model** with new lead data to maintain performance over time, especially as customer behavior evolves.  
- **Use model insights for personalized sales strategies**, focusing efforts on leads with high predicted conversion probabilities and tailoring engagement based on lead profile attributes.

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

