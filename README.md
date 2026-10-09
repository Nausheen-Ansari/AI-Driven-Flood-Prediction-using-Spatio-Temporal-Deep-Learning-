# AI-Driven-Flood-Prediction-using-Spatio-Temporal-Deep-Learning-
A Spatio-Temporal Deep Learning pipeline using ConvLSTM2D and a custom Dice Loss function to forecast pixel-wise flood risk maps from sequential geographical rainfall data.

An end-to-end AI pipeline for advanced flood risk forecasting. This project overcomes the limitations of standard time-series models (which lack geographical awareness) and traditional Convolutional Neural Networks (which lack temporal memory) by utilizing a **ConvLSTM2D** architecture to process fluid weather systems across both space and time.

## 📌 Project Overview

Floods are highly dynamic natural disasters driven by complex, moving weather systems. This project processes raw meteorological coordinates and historical disaster records, mathematically transforms them into continuous 3D spatial grids, and trains a deep learning model to forecast pixel-wise flood risk maps. A custom optimization strategy is implemented to solve the severe class imbalance between normal dry land and flooded regions.

## 🚀 Key Features

* **Spatial Data Engineering:** Transforms discrete 1D rainfall coordinates (Latitude/Longitude) into structured 32x32 continuous spatial matrices using `scipy.interpolate.griddata`.
* **Spatio-Temporal Memory:** Ingests a 3-day temporal sequence (T-3 to T-1) of spatial rainfall grids to predict the exact geographical flood risk for the following day (T+1).
* **Custom Optimization (Dice Loss):** Solves extreme pixel-class imbalance. Instead of standard binary cross-entropy (which heavily biases toward predicting dry land), the model utilizes a custom Dice Loss function integrated directly into the TensorFlow graph to optimize exclusively for overlapping flood zones.
* **Pixel-Wise Evaluation:** Evaluates model performance using spatial confusion matrices, classification reports (Precision/Recall/F1), and 1x5 visual prediction grids.

## 🧠 Architecture & Methodology

1. **Data Alignment:** Synchronizes the CWC daily rainfall coordinate dataset with dates from the India Flood Inventory.
2. **Target Masking:** Applies an 85th percentile threshold to rainfall on known flood days to generate binary `1.0` (Flood) and `0.0` (No Flood) ground truth masks.
3. **Model Core:**
* `Input(3, 32, 32, 1)`
* `ConvLSTM2D` (32 filters, 3x3 kernel, temporal feature extraction)
* `BatchNormalization` & `Dropout(0.2)` (Regularization)
* `Conv2D` (1 filter, sigmoid activation, final spatial risk mapping)



## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Deep Learning:** TensorFlow / Keras
* **Data Manipulation:** Pandas, NumPy
* **Scientific Computing:** SciPy (Spatial Gridding & Interpolation)
* **Visualization:** Matplotlib, Seaborn

## 📂 Dataset Requirements

To run this pipeline, two datasets are required in the root directory:

1. `51acd4ce-fb56-43f5-9a0b-ea621a1b9683.csv` (CWC Manual Daily Rainfall)
2. `India_Flood_Inventory_v3.csv` (Historical Flood Event Dates)

## 💻 Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/spatio-temporal-flood-prediction.git
cd spatio-temporal-flood-prediction

```

2. Install required dependencies:

```bash
pip install tensorflow pandas numpy scipy matplotlib seaborn scikit-learn

```

3. Launch the Jupyter Notebook or execute the training script:

```bash
jupyter notebook "ConvoLSTM Prototype_.ipynb"

```

## 🔍 The Dice Loss Function

To address the overwhelming ratio of dry land to flooded pixels, the network is compiled using this custom loss function, forcing the optimizer to maximize the mathematical overlap of the minority class:

```python
import tensorflow as tf

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - (numerator + 1) / (denominator + 1)

```

## 📊 Results & Visualization

The pipeline automatically generates diagnostic visualizations upon completion of the training loop:

* **Training Diagnostics:** Epoch-by-epoch Binary Crossentropy Loss and Accuracy curves.
* **Spatial Confusion Matrix:** Highlighting pixel-wise True Positives (Actual Floods correctly predicted).
* **Forecast Maps:** A 1x5 Spatio-Temporal forecast grid comparing the input rainfall sequence, the actual ground truth flood map, and the AI's predicted risk heatmap.

## 👨‍💻 Author

**Nausheen Ansari**

* B.Tech Computer Science and Engineering (Artificial Intelligence & Machine Learning)
* Manav Rachna International Institute of Research and Studies
