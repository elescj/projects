# 🏀 Winning in the Small Ball Era (2010–2025): a Regression Analysis
This project applies Linear Regression, Lasso Regression, Decision Tree, and Random Forest to forecast winning percentage of an NBA team, providing actionable GM recommendations using the best-performing regression model.
![Graphical Summary](attachments/small-ball.png)

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
This project implements a supervised machine learning pipeline to predict an NBA team's winning percentage, leveraging multiple regression methods to model the relationship between win/lose rate and features such as possession, points per possession, and shot types. The pipeline includes data cleaning, feature engineering, model training, evaluation, and insights extraction to build a successful team.

## 📊 Dataset
These datasets, originally provided on Basketball-Reference.com, is a set of four basketball team season statistics in 15 years, where each row represents a team's season statistics, including details such as minutes played, points, rebounds, and assists.

After data preprocessing, the table set is merged into one operatable dataset. Each record includes **25 input features** describing team performance, and one **target variable**: Win/Lose Rate.

Detailed feature descriptions are listed in the table below:
## Data Dictionary
| Variable | Description |
|----------|-------------|
| Rk       | Team rank in the season |
| Team     | Name of the NBA team |
| G        | Games played |
| MP       | Minutes played |
| FG       | Field goals made |
| FGA      | Field goals attempted |
| FG%      | Field goal percentage (FG ÷ FGA) |
| 3P       | Three-point field goals made |
| 3PA      | Three-point field goals attempted |
| 3P%      | Three-point field goal percentage (3P ÷ 3PA) |
| 2P       | Two-point field goals made |
| 2PA      | Two-point field goals attempted |
| 2P%      | Two-point field goal percentage (2P ÷ 2PA) |
| FT       | Free throws made |
| FTA      | Free throws attempted |
| FT%      | Free throw percentage (FT ÷ FTA) |
| ORB      | Offensive rebounds |
| DRB      | Defensive rebounds |
| TRB      | Total rebounds (ORB + DRB) |
| AST      | Assists |
| STL      | Steals |
| BLK      | Blocks |
| TOV      | Turnovers |
| PF       | Personal fouls |
| PTS      | Total points scored |

## ❓ Problem Statement
This project aims to uncover the winning formula of the small-ball era by analyzing regular-season team-level statistics from 2010 to 2025. Rather than focusing on individual players or limited head-to-head matchups, we use season-level metrics to identify the key factors driving strong win–loss records.  

Regular-season data provides a stable and efficient measure of team performance, capturing the cumulative impact of opponents and game contexts. Our goal is to understand which scoring and efficiency metrics most influence winning, offering actionable insights for team building and roster construction in the small-ball era.

## 🔎 Methodology
The following steps outline the end-to-end process used in this project:

1. **Data Pipeline**
   - **Data Preprocessing**: Construct an operatable dataset from the original URL.
   - **Data Overview**: Reviewed to understand data types, check for duplicates, and assess missing values.
   - **Exploratory Data Analysis (EDA)**: **Univariate Analysis** and **Bivariate Analysis** explored the data distribution and variable relations.
   - **Feature Engineering**: Transformed vague or inconsistent variables into more meaningful, usable forms for modeling (e.g., number of posession, creating `POS`).

2. **Modelling**
   - Performed **Linear Regression, Lasso Regression, Decision Tree, and Random Forest** separately to forecast used car prices.
   - Compared **MSE, RMSE, MAE, R²** across all models to evaluate and select the best-performing method for price prediction.

## 📈 Results

| Model              | MSE       | RMSE      | MAE       | R²       |
|-------------------|-----------|-----------|-----------|----------|
| Lasso             | 0.000767  | 0.027695  | 0.021497  | 0.970058 |
| Random Forest     | 0.001042  | 0.032276  | 0.025863  | 0.959331 |
| Linear Regression | 0.001085  | 0.032940  | 0.025275  | 0.957643 |
| Decision Tree     | 0.001237  | 0.035174  | 0.028044  | 0.951701 |

**Lasso Regression achieves the highest Test R² (`0.970`) and the lowest Test MSE, RMSE, and MAE**, indicating **best generalization** and **predictive accuracy** on unseen data.

## 💡 Insights & Recommendations

- Generating high-value three-point attempt.  
- Attacking the basket to pressure interior defenses.
- Defending the three-point line aggressively.
- Forcing opponents into lower-efficiency mid-range shots.

<a id="technologies-used"></a>
## ⚙️ Technologies Used
- **Python** – General purpose programming
- **Pandas** – Data manipulation and analysis
- **SciPy** – Statistical tests and probability distributions
- **NumPy** – Numerical computations
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Model training and evaluation

<a id="how-to-run"></a>
## ▶️ How to Run
```bash
# Clone the repository
git clone https://github.com/elescj/013-small-ball-lr.git
cd 013-small-ball-lr

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the script
python main.py
```
