# 🎬 Movie Recommendation Systems: A Comparative Study
This project implements four recommendation system models to predict users’ movie preferences. It evaluates model performance using metrics like Precision@K, Recall@K, and F₁-score to generate personalized top-N recommendations.
![Graphical Summary](attachments/movies.png)

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
This project implements four recommendation approaches: rank-based (average ratings), user–user similarity, item–item similarity, and model-based collaborative filtering (matrix factorization). Model performance is evaluated using Precision@K, Recall@K, and F₁-score, enabling the generation of personalized top-N movie recommendations.

## 📊 Dataset
This dataset was originally provided as part of the **Applied Data Science Program by MIT**. It is a **100,836 × 4 CSV file**, where each row represents a user’s rating of a movie. Each record contains four features describing **which user**, **which movie**, **the rating**, and **when the rating was made**.  

| Variable   | Description                         |
|------------|-------------------------------------|
| userId     | Unique identifier for each user     |
| movieId    | Unique identifier for each movie    |
| rating     | Rating given by the user to the movie |
| timestamp  | Time when the rating was recorded   |

## ❓ Problem Statement
Movie streaming platforms offer thousands of titles, but users often struggle to discover movies that match their preferences. Traditional word-of-mouth recommendations are limited by social connections and subjective opinions. 

The goal of this project is to **build a recommendation system** that can predict a user's movie preferences and provide **personalized top-N recommendations**. This involves implementing and comparing multiple approaches, including **rank-based methods, collaborative filtering (user-user and item-item), and model-based matrix factorization**, while evaluating their performance using metrics like **Precision@K, Recall@K, F₁-score, and RMSE**.

## 🔎 Methodology
### Methodology

The recommendation system development followed an **end-to-end workflow** from data preprocessing to model evaluation and delivery:

1. **Data Preparation**  
   - Loaded the dataset containing user–movie ratings.  
   - Performed basic exploratory data analysis to understand rating distributions, user activity, and movie popularity.  
   - Created a **user–item interaction matrix** for collaborative filtering models.

2. **Model Implementation**  
   - **Rank-based recommendation:** Predicted ratings based on average movie ratings.  
   - **User–User collaborative filtering:** Estimated ratings using similarity between users.  
   - **Item–Item collaborative filtering:** Estimated ratings using similarity between items.  
   - **Model-based collaborative filtering (SVD):** Factorized the user–item matrix to capture latent features.

3. **Model Evaluation**  
   - Split data into training and test sets.  
   - Evaluated models using **Precision@K, Recall@K, F₁-score, and RMSE** to measure recommendation accuracy and coverage.  

4. **Hyperparameter Optimization**  
   - Applied **grid search cross-validation** to identify optimal parameters for similarity-based and SVD models.  
   - Selected the best-performing models for generating final recommendations.

5. **Recommendation Delivery**  
   - Generated **top-N personalized movie recommendations** for each user based on predicted ratings.  
   - Optionally, ranked recommendations using **corrected ratings** that account for both predicted ratings and movie popularity.

This methodology ensures a **robust, end-to-end pipeline** for building and evaluating movie recommendation systems.

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
