# 🔊Audio MNIST Digit Recognition
This project applies Convolutional Neural Network (CNN) models to classify food images into their respective categories, with the objective of developing the most accurate and reliable CNN architecture for food recognition.
![Graphical Summary](attachments/food-picture.png)

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
This project implements a **deep learning pipeline** for **classifying food images** into their respective categories. Using **Convolutional Neural Networks (CNNs)**, the model learns visual patterns that capture the unique characteristics of each food type, allowing it to accurately classify previously unseen (unlabeled) images.

The pipeline includes:
- **Data preprocessing and image handling**
- **Design and implementation of two CNN architectures**
- **Model training, evaluation, and performance comparison**
- **Demonstration of the best-performing model on sample images**

## 📊 Dataset

The dataset for this project is sourced from the **MIT Applied Data Science Program** and is organized as follows:

- The dataset is provided in **ZIP format**, containing **training** and **test** folders.
- Each folder has **three subfolders**, each representing a **food category**.
- All images are in **JPG format**.

> **Note:** The original dataset exceeds GitHub's upload limit.  
> To access the data, please contact me via my portfolio website:  
> [Contact Charles Jiao](https://charles-jiao.netlify.app/contact)


## ❓ Problem Statement
Classifying food categories from images is important for applications such as automated dietary tracking, restaurant menu organization, and food recommendation systems. It’s a challenging problem because food images can vary widely in appearance due to differences in presentation, lighting, angle, and portion size. In this project, we use a labeled dataset of food images to build convolutional neural network (CNN) models that classify images into their respective food categories, providing practical insights into visual recognition of foods.

## 💻 Methodology
The following steps outline the end-to-end process used in this project:

1. **Data (image) Processing**
   - **Data Retrieval and Preparation**: Extract data from the ZIP file and prepare the training and test data.
   - **Data Visualization**: Randomly choose a few images and observe their potential features that may be extracted in CNN.
   - **Data Preprocessing**: Label the data, split into training and test set, normalize the images, and encode the categories.

2. **Modelling**
   - Build two models with different neuron structures for comparison.
   - For each model, build the architecture, training this model, and evaluate the model accuracy and confusion matrix.
   - Run a prediction with the better model.

## 📝 Results

The models were evaluated on **test** datasets using standard classification metrics.  

- **Precision (1)**: When the model predicts a conversion, how often it is correct.  
- **Recall (1)**: Out of all actual conversions, how many were correctly identified.  
- **F1-Score (1)**: Balance between precision and recall.  
- **Accuracy**: Overall correct predictions (can be misleading if classes are imbalanced).  

**Classification Report**

| Class | Precision | Recall | F1-Score | Support |
|-------|----------|--------|----------|---------|
| 0     | 0.77     | 0.65   | 0.71     | 362     |
| 1     | 0.84     | 0.87   | 0.85     | 500     |
| 2     | 0.77     | 0.89   | 0.82     | 232     |

**Accuracy:** 0.80  
**Macro Avg:** Precision: 0.79, Recall: 0.80, F1-Score: 0.79  
**Weighted Avg:** Precision: 0.80, Recall: 0.80, F1-Score: 0.80

## 💡 Insights & Recommendations

### 🔎 Insights
In this project, we built a Convolutional Neural Network (CNN) to classify food images into three categories: **Bread**, **Soup**, and **Vegetable-Fruit**.  

The model achieved an overall test accuracy of **66%**, with class-wise performance varying significantly:

- **Vegetable-Fruit**: Strong performance (F1-score 0.82), correctly classified most images.  
- **Soup**: Moderate performance (F1-score 0.72) with high recall (0.90), indicating most Soup images were correctly identified.  
- **Bread**: Weak performance (F1-score 0.35), frequently misclassified as Soup, showing challenges in distinguishing visually similar categories.  

Confusion matrix analysis confirmed that the model struggles most with **Bread vs. Soup** due to overlapping visual features or possible dataset imbalance.  

Overall, the model demonstrates moderate capability in classifying food images, performing better on distinct classes (**Vegetable-Fruit**) than on similar ones (**Bread** and **Soup**).

### ✅ Recommendations
1. **Increase Data Variety and Augmentation**  
   - Add more Bread images or apply data augmentation (rotation, flipping, scaling) to reduce misclassification.

2. **Address Class Imbalance**  
   - Use class weighting or oversampling techniques to give the model more emphasis on underrepresented classes.

3. **Use Transfer Learning**  
   - Pre-trained models like **VGG16**, **ResNet**, or **EfficientNet** could improve feature extraction, especially for visually similar classes.

4. **Hyperparameter Tuning**  
   - Experiment with different architectures, optimizers, learning rates, batch sizes, and dropout rates to improve generalization.

5. **Additional Evaluation Metrics**  
   - Consider precision-recall curves, ROC-AUC, or top-2 accuracy to better understand model performance on ambiguous cases.

6. **Deployment Considerations**  
   - For real-world use in a stock photography platform, implement a **human-in-the-loop verification** for low-confidence predictions to ensure labeling quality.

<a id="technologies-used"></a>
## ⚙️ Technologies Used
- **Python** – General purpose programming
- **NumPy** – Numerical computations
- **Pandas** – Data manipulation and analysis
- **OpenCV** – Image processing and augmentation
- **Matplotlib & Seaborn** – Data and performance visualization
- **Scikit-learn** – Data preprocessing, model evaluation, and metrics
- **TensorFlow & Keras** – Building, training, and deploying Convolutional Neural Networks (CNN)
- **Warnings** – For suppressing non-critical output

<a id="how-to-run"></a>
## ▶️ How to Run
```bash
# Clone the repository
git clone https://github.com/elescj/007-food-picture-lr.git
cd 007-food-image-lr

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the script
python main.py
```

