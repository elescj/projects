# 🏘️ Using CNN to Recognize Street View Housing Number
This project builds a multi-class image classification model to predict 10 object categories from the CIFAR-10 dataset using Convolutional Neural Networks (CNNs) and Transfer Learning, with the objective of improving accuracy and training efficiency on small image datasets.
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
This project implements a **deep learning pipeline** for **classifying images** into their respective categories. Using **Convolutional Neural Networks (CNNs)**, the model learns visual patterns that capture the unique characteristics of each food type, allowing it to accurately classify previously unseen (unlabeled) images; using **Transfer Learning**, the pre-trainrf model processing new images with higher productivity.

The pipeline includes:
- **Data preprocessing and image handling**
- **Design and implementation of three CNN architectures**
- **Model training, evaluation, and performance comparison**
- **Transfer learning and prediction**

## 📊 Dataset

The CIFAR-10 dataset consists of 60000 32x32x3, i.e., color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
> [Learn more about the data](https://www.cs.toronto.edu/~kriz/cifar.html)


## ❓ Problem Statement
Classifying images is important for applications such as computer vision, automation, and robotics. It’s a challenging problem because images can vary widely in appearance due to differences in presentation, lighting, angle, and portion size. In this project, we use the CIFAR-10 dataset (Canadian Institute for Advanced Research) to build convolutional neural network (CNN) models that classify images into their respective categories, providing a pre-trained model that is later applied in transfer learning for vast image classification application.

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

3. **Transfer Learning**
   - Build one transfer layer.
   - Compile and fit the model.

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

