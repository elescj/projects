# 🏘️ Using CNN to Recognize Street View Housing Number
This project builds and compares two convolutional neural net work  (CNN) models to classify digits (0-9) from stree-view house numbers using the SVHN dataset. The better-performing model is then used to make predictions on unseen images, demonstrating the effectiveness of CNNs for real-world digit recognition.
![Graphical Summary](attachments/street-view.png)

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
This project implements a **deep learning pipeline** to **classify written digits (0-9) from stree-view house numbers** using **Convolutional Neural Networks (CNNs)**. The model learns visual patterns that capture the unique characteristics of each digit, enabling accurate classification of previously unseen (unlabeled) images.

The pipeline includes:
- **Data preprocessing and image handling**
- **Design and implementation of two CNN architectures**
- **Model training, evaluation, and performance comparison**
- **Prediction on the test data and assessment of model accuracy**

## 📊 Dataset

The SVHN dataset contains over than 600,000 labeled digits images extracted from real-world street-view house numbers. Each sample is a 32×32 (RGB) image, providing consistent dimensions ideal for training deep learning models. The dataset is divided into three primary subsets:

- Training set: 73,257 images
- Test set: 26,032 images
- Extra set: 531,131 images for extended training

## ❓ Problem Statement
Classifying images is essential in computer vision, automation, and robotics, it remains challenging variations in lighting, orientation, and overall image quality. In this project, I used the SVHN dataset to develop CNN-based classification models capable of accurately identifying digit images. The project produces a trained deep learning model along with a full training-evaluation pipeline that supports downstream prediction tasks.

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
|-------|----------|--------|----------|----------|
| 0     | 0.83     | 0.83   | 0.83     | 1000     |
| 1     | 0.85     | 0.89   | 0.87     | 1000     |
| 2     | 0.75     | 0.67   | 0.71     | 1000     |
| 3     | 0.58     | 0.66   | 0.61     | 1000     |
| 4     | 0.74     | 0.76   | 0.75     | 1000     |
| 5     | 0.63     | 0.70   | 0.66     | 1000     |
| 6     | 0.87     | 0.79   | 0.83     | 1000     |
| 7     | 0.84     | 0.82   | 0.83     | 1000     |
| 8     | 0.83     | 0.91   | 0.87     | 1000     |
| 9     | 0.93     | 0.75   | 0.83     | 1000     |

**Accuracy:** 0.78 
**Macro Avg:** Precision: 0.78, Recall: 0.78, F1-Score: 0.78  
**Weighted Avg:** Precision: 0.78, Recall: 0.78, F1-Score: 0.78

## 💡 Insights & Recommendations

### 🔎 Insights
In this project, we built a Convolutional Neural Network (CNN) to classify images on the CIFAR-10 dataset.  

We have seen four different iterations of the CNN model and built an intuition about how to improve the model by tuning various hyperparameters and using different techniques. There is still plenty of scope for improvement and you can try out tuning different hyperparameters to improve the model performance.

### ✅ Recommendations
1. **Tuning hyperparameters**  
   - Tune hyper parameters to improve model performance.

2. **Increasing CNN layers**  
   - Increase the layer number to improve prediction accuracy.

<a id="technologies-used"></a>
## ⚙️ Technologies Used
- **Python** – General purpose programming
- **NumPy** – Numerical computations
- **Pandas** – Data manipulation and analysis
- **Matplotlib & Seaborn** – Data and performance visualization
- **Scikit-learn** – Data preprocessing, model evaluation, and metrics
- **TensorFlow & Keras** – Building, training, and deploying Convolutional Neural Networks (CNN)

<a id="how-to-run"></a>
## ▶️ How to Run
```bash
# Clone the repository
git clone https://github.com/elescj/009-image-classification-lr.git
cd 009-image-classification-lr

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the script
python main.py
```

