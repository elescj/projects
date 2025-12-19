# 😀 Facial Emotion Recognition: CNNs and Transfer Learning for Grayscale vs RGB
This project explores facial emotion recognition using deep learning. Multiple convolutional neural network models were trained and evaluated on grayscale and RGB images to analyze the impact of color representation on model performance. The study further investigates transfer learning techniques to improve accuracy and generalization, with comprehensive evaluation using accuracy, loss, and confusion matrices.
![Graphical Summary](attachments/facial.png)

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
This project implements a **deep learning pipeline** to **classify facial emotions (happy, sad, neutral, surprise)** from facial images using **Convolutional Neural Networks (CNNs)**. The models learn discriminative visual features from facial expressions, enabling accurate classification of previously unseen (unlabeled) images.

The pipeline includes:
- Data preprocessing and image handling (grayscale and RGB)
- Design and implementation of multiple CNN architectures
- Model training, evaluation, and performance comparison
- Transfer learning using a pre-trained **VGG16** model

## 📊 Dataset

The dataset for this project is sourced from the **MIT Applied Data Science Program** and is organized as follows:

- The dataset is provided in **ZIP format**, containing **training** and **test** folders.
- Each folder has **two subfolders**, each representing a **health condition of the cell**.
- All images are in **PNG format**.

> **Note:** The original dataset exceeds GitHub's upload limit.  
> To access the data, please contact me via my portfolio website:  
> [Contact Charles Jiao](https://charles-jiao.netlify.app/contact)

## ❓ Problem Statement
Classifying images is essential in computer vision, automation, and robotics, it remains challenging in fields such as public health where high accuracy makes great significance. In this project, I used the microscopic images to develop CNN-based classification models capable of accurately identifying blood cell images. The project produces a trained deep learning model along with a full training-evaluation pipeline that supports downstream prediction tasks.

## 💻 Methodology
The following steps outline the end-to-end process used in this project:

1. **Data (image) Processing**
   - **Data Retrieval and Preparation**: Extract data from the ZIP file and prepare the training and test data.
   - **Data Demonstration**: Randomly choose a few images and observe their potential features that may be extracted in CNN; demonstrate the class distribution.
   - **Data Preprocessing**: Label the data, split into training and test set, normalize the images, and encode the categories.

2. **Modelling**
   - Compare basic neural network and CNN models' performance applied in greyscale and RBG.
   - Build two transfer learning models for comparison.
   - The models include an initial base model, refined model with more neuron layers and augmented data.
   - For each model, build the architecture, training this model, and evaluate the model accuracy.

3. **Transfer Learning**
   - Build two transfer learning models (VGG16 and ResNet V2 models).
   - For each model, build the architecture, training this model, and evaluate the model accuracy.

4. **Final Model**
   - Apply the best model on the dataset.
   - Evaluate the model accuracy and confusion matrix.

## 📝 Results

### Overall Comparison

The models were evaluated on the test set using **accuracy, loss, and confusion matrices**. Both grayscale and RGB images were compared across multiple CNN architectures and transfer learning approaches. A quick overview at different models' accuracy.

| Model / Technique           | Input Mode | Accuracy | Remarks                                               |
|-----------------------------|------------|---------|------------------------------------------------------|
| Base CNN (Model 1)          | Grayscale  | 0.68    | Simple, stable, performs better on grayscale        |
| Base CNN (Model 1)          | RGB        | 0.66    | Slightly worse, additional channels not useful      |
| Advanced CNN (Model 2)      | Grayscale  | 0.76    | Best performance among CNNs, suitable for grayscale|
| Advanced CNN (Model 2)      | RGB        | 0.30    | RGB caused significant drop; model overfits         |
| Complex CNN (Model 3)       | Grayscale  | 0.50    | Overly deep, moderate performance, potential overfitting |
| VGG16 Transfer Learning     | RGB        | 0.25    | Poor performance, pre-trained features not compatible |
| ResNet50 TL                 | RGB        | 0.25    | Similar to VGG16, unsuitable for small emotion dataset |
| EfficientNet TL             | RGB        | 0.25    | Small dataset and grayscale features not captured   |

### Final Model

- **Precision (1)**: When the model predicts a conversion, how often it is correct.  
- **Recall (1)**: Out of all actual conversions, how many were correctly identified.  
- **F1-Score (1)**: Balance between precision and recall.  
- **Accuracy**: Overall correct predictions (can be misleading if classes are imbalanced).  

**Classification Report**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.66      | 0.84   | 0.74     | 32      |
| 1     | 0.60      | 0.56   | 0.58     | 32      |
| 2     | 0.63      | 0.59   | 0.61     | 32      |
| 3     | 0.89      | 0.75   | 0.81     | 32      |

**Overall Metrics**

- Accuracy: 0.69  
- Macro Average: Precision = 0.70, Recall = 0.69, F1-Score = 0.69  
- Weighted Average: Precision = 0.70, Recall = 0.69, F1-Score = 0.69  

**Confusion Matrix/ Heatmap**

![Confusion Matrix](attachments/confusion-matrix.png)

**Accuracy vs. Epoch**

![Accuracy vs. Epoch](attachments/accuracy.png)

## 💡 Insights & Recommendations

### Insights
**Impact of Input Color Mode**
- Models trained on **grayscale images** generally performed better than those trained on **RGB images** for this dataset.  
- **Base CNN models** achieved ~68–76% accuracy on grayscale but dropped significantly (~29–66%) on RGB, suggesting that additional color channels did not improve feature learning for facial emotion detection.

**Model Complexity vs Performance**
- Increasing model complexity (Model 3 – 5 CNN blocks) did **not guarantee better performance**.  
- Overly deep architectures showed **overfitting** or unstable training, especially on small datasets or with grayscale inputs.  
- Simpler models (Model 1 and Model 2) with appropriate regularization and batch normalization were more **stable** and often outperformed complex CNNs.

**Transfer Learning Architectures**
- Pre-trained models like **VGG16, ResNet50, EfficientNet** did not yield satisfactory performance (~25% accuracy) on this dataset.  
- Possible reasons:  
  - These models expect 3-channel RGB images, making them incompatible with grayscale inputs without modification.  
  - Small dataset limits fine-tuning capability.  
  - Domain mismatch: pre-trained weights were trained on general images (ImageNet), which may not capture subtle facial emotion features.

**Observations from Confusion Matrices**
- Certain classes (e.g., **“happy”** or **“surprise”**) were predicted better than others, indicating **class-specific bias**.  
- Models struggle with subtle emotions like **“neutral”** or **“sad”**, which may have overlapping facial features.

### Future Work

**Input Representation**
- Use **grayscale images** with standardized size (48×48) to simplify the model and reduce computational overhead.  
- Optionally, apply **histogram equalization** or **normalization** for enhanced feature contrast.

**Model Architecture**
- **Advanced CNN (Model 2)** is recommended as the base:  
  - 3 convolutional blocks with `Conv2D → BatchNormalization → LeakyReLU → MaxPooling → Dropout`.  
  - Flatten → Dense layers (512 → 128) → Softmax.  
- Keep the network **shallow enough** to avoid overfitting while capturing meaningful spatial features.

**Training Strategy**
- Use **data augmentation**: horizontal flips, slight brightness adjustment, shear transforms.  
- Apply **early stopping** and **learning rate reduction** to stabilize training.  
- Train for **20–30 epochs** with a moderate batch size for optimal performance.

<a id="technologies-used"></a>
## ⚙️ Technologies Used
- **Python** – General purpose programming
- **NumPy** – Numerical computations
- **Matplotlib & Seaborn** – Data and performance visualization
- **Scikit-learn** – Data preprocessing, model evaluation, and metrics
- **TensorFlow & Keras** – Building, training, and deploying Convolutional Neural Networks (CNN)
- **OpenCV** – Image processing and augmentation

<a id="how-to-run"></a>
## ▶️ How to Run
```bash
# Clone the repository
git clone https://github.com/elescj/011-malaria.git
cd 011-malaria-lr

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the script
python main.py
```
