# 🚗 Benchmarking Regression Models on Indian Used Car Price Prediction
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

📌 Note: The Boston Housing dataset has been deprecated in newer versions of scikit-learn due to ethical concerns about the LSTAT variable. It is used here solely for educational purposes.

## ❓ Problem Statement
Accurately estimating housing prices is a persistent challenge due to the numerous and often interrelated factors that influence real estate markets — particularly the issue of multicollinearity among predictors. This project aims to develop a linear regression model to predict housing prices in Boston using key features from the dataset. The objective is to minimize prediction error while identifying the most influential variables, offering a foundational, interpretable baseline for housing price modeling and further machine learning applications.

## 🔎 Methodology
The following steps outline the end-to-end process used in this project:

1. **Data Overview**  
   The Boston Housing dataset was loaded into a pandas DataFrame for preliminary inspection. Key characteristics such as data dimensions, data types, presence of duplicates or null values, and the number of unique values per feature were assessed.
   
3. **Exploratory Data Analysis (EDA)**  
   - **Univariate analysis** was conducted to understand the distribution of each feature using histograms and descriptive statistics.  
   - **Bivariate analysis** explored relationships between independent variables and the target (`MEDV`) using scatter plots and correlation heatmaps, helping to identify potential multicollinearity and feature relevance.

4. **Model Building – Linear Regression**  
   - Split the dataset into training and testing sets (typically 80/20).  
   - Examined multicollinearity using correlation matrices and VIF; dropped highly collinear features.  
   - Built an initial linear regression model and evaluated coefficient significance.  
   - Removed statistically insignificant variables to improve model interpretability and re-fit the model.  
   - Verified linear regression assumptions (linearity, normality, homoscedasticity, independence).  
   - Evaluated model performance using **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, and **Mean Absolute Percentage Error (MAPE)**.  
   - Applied **k-fold cross-validation** to assess model generalizability.  
   - Constructed the final linear regression model based on refined features and evaluation metrics.

## 📈 Results

### 🔢 Model Performance Metrics
The linear regression model was evaluated using standard regression metrics on the test set:
| Metric                                    | Value             |
|-------------------------------------------|-------------------|
| R² Score (Coefficient of Determination)   | 0.729 (+/- 0.232) |
| RMSE (Root Mean Squared Error)            | 0.198045          |
| MAE (Mean Absolute Error)                 | 0.151284          |
| MAPE (Mean Absolute Percentage Error)     | 5.257965          |

These results indicate that the model explains approximately 73% of the variance in housing prices and achieves a mean absolute percentage error of approximately 5.26%.

---

### 📊 Model Coefficients
Below are the learned coefficients for each feature in the model:
| Feature   | Coefficient | Interpretation (qualitative impact)                   |
|-----------|-------------|--------------------------------------------------------|
| const     | 4.649       | Baseline value when all features are 0                |
| CRIM      | -0.0125     | Higher crime rate slightly decreases housing price    |
| CHAS      | +0.1198     | Proximity to Charles River slightly increases price   |
| NOX       | -1.0562     | Higher air pollution (NOx) strongly decreases price   |
| RM        | +0.0589     | More rooms per dwelling increases price               |
| DIS       | -0.0441     | Greater distance from employment centers reduces value|
| RAD       | +0.0078     | Higher accessibility to highways slightly increases price |
| PTRATIO   | -0.0485     | Higher student–teacher ratio reduces price            |
| LSTAT     | -0.0293     | Higher % of lower status population decreases price   |

*Note: The magnitude and sign of coefficients indicate each feature’s relative impact.*

---

### 🧪 Residual Analysis
To assess the model assumptions and quality of fit, the following plots were analyzed:
- ✅ **Residuals vs. Fitted Values** (for linearity & homoscedasticity)
- ✅ **Histogram of Residuals** (for normality)
- ✅ **QQ Plot** (for distribution shape)
- ✅ **Actual vs. Predicted** plot (for overall fit)

![Residuals vs. Fitted Values](attachments/residual.png)
![Histogram of Residuals](attachments/histogram.png)
![QQ Plot](attachments/qq-plot.png)

**Example code (matplotlib)**
```python
# Plot histogram of residuals.
sns.histplot(residuals, kde=True)
# Show the plot.
plt.show()
# Plot q-q plot of residuals
stats.probplot(residuals, dist = "norm", plot = pylab)
# Show the plot.
plt.show()
```

## 💡 Insights & Recommendations
- Actionable insights have been derived for each key variable influencing housing prices.
- Based on the model results, we provide the following actionable insights for various stakeholders, including urban planners, policymakers, real estate developers, homeowners, and investors.

<a id="technologies-used"></a>
## ⚙️ Technologies Used
- **Python** – General purpose programming
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computations
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Model training and evaluation
- **Statsmodels** – Statistical modeling (OLS regression, VIF)
- **SciPy** – Statistical tests and probability distributions
- **Pylab** – Numerical plotting and analysis
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
