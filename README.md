
# 🌾 Crop Yield and Field Prediction using Machine Learning

A simple Python application with a GUI (built using Tkinter) that predicts crop production and yield based on input agricultural data. The project utilizes machine learning (Decision Tree Regressor) to train on crop datasets and make predictions on unseen data.

## 🚀 Features

- Upload and preview crop production datasets (CSV format).
- Preprocess and encode categorical features using LabelEncoder.
- Train a Decision Tree Regression model.
- Upload test data and predict crop yield (in tones/acre) and total production (in KGs).
- User-friendly GUI built with Tkinter.

## 🛠️ Libraries Required

- `numpy`
- `pandas`
- `sklearn`
- `tkinter` *(comes pre-installed with Python)*

## 📁 Project Structure

```
CropYieldPrediction/
│
├── field_prediction.py               # The main Python GUI application
├── Dataset/              # Folder to store your CSV files (training and test)
├── README.md             # Project documentation (this file)
```

## ✅ Steps to Run the Project

1. **Install required libraries**

```bash
pip install numpy pandas scikit-learn
```

> Note: `tkinter` is built-in with Python. If it's missing, you can install via:
> - On Ubuntu: `sudo apt-get install python3-tk`
> - On Windows: Make sure it's included during Python installation.

2. **Run the Application**

```bash
python main.py
```

3. **Use the GUI to:**

   - Upload a crop dataset (`CSV`) with columns: `State_Name, District_Name, Season, Crop, Year, Area, Production`
   - Preprocess the dataset (LabelEncoding and normalization)
   - Train the ML model
   - Upload test dataset and view predictions

## 📊 Sample Output

```
Test Record 1: Production = 1200.00 KGs
Yield = 1.20 tones/acre
```

## 🧠 Model Info

- **Algorithm**: Decision Tree Regressor
- **Parameters**: `max_depth=100`, `max_leaf_nodes=20`, `max_features=5`, `splitter='random'`
- **Evaluation Metric**: Root Mean Squared Error (RMSE)
