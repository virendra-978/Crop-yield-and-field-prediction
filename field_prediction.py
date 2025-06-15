from tkinter import *
from tkinter.filedialog import askopenfilename
import tkinter
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

main = tkinter.Tk()
main.title("Crop Yield and Field Prediction")
main.geometry("1350x850")
main.config(bg="#f0f0f5")

# Globals
filename = ""
X_train = X_test = y_train = y_test = X = Y = dataset = le = model = None
X_train = X_test = y_train = y_test = X = Y = dataset = None
model = None
le_state = le_district = le_season = le_crop = None

# Upload CSV
def upload():
    global filename, dataset
    text.delete('1.0', END)
    filename = askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace=True)
    dataset['Production'] = dataset['Production'].astype(np.int64)
    text.insert(END, '✅ Dataset uploaded successfully!\n\n')
    text.insert(END, str(dataset.head(10)) + "\n")

# Preprocess Data
def processDataset():
    global le_state, le_district, le_season, le_crop
    global dataset, X_train, X_test, y_train, y_test, X, Y
    text.delete('1.0', END)

    le_state = LabelEncoder()
    le_district = LabelEncoder()
    le_season = LabelEncoder()
    le_crop = LabelEncoder()

    dataset['State_Name'] = le_state.fit_transform(dataset['State_Name'])
    dataset['District_Name'] = le_district.fit_transform(dataset['District_Name'])
    dataset['Season'] = le_season.fit_transform(dataset['Season'])
    dataset['Crop'] = le_crop.fit_transform(dataset['Crop'])

    text.insert(END, '✅ Data preprocessed.\n\n')
    text.insert(END, str(dataset.head(10)) + "\n")

    data = dataset.values
    X = normalize(data[:, :-1])
    Y = data[:, -1].astype('uint8')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.insert(END, f"\nTotal Records: {len(X)}\n")
    text.insert(END, f"Training Records: {X_train.shape[0]}\n")
    text.insert(END, f"Testing Records: {X_test.shape[0]}\n")

# Train Model
def trainModel():
    global model, X_train, X_test, y_train, y_test, X, Y
    text.delete('1.0', END)

    model = DecisionTreeRegressor(max_depth=100, random_state=0, max_leaf_nodes=20, max_features=5, splitter="random")
    model.fit(X, Y)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(preds, y_test)) / 1000

    text.insert(END, "✅ Model training complete.\n")
    text.insert(END, f"Decision Tree RMSE: {rmse:.4f}\n")

# Predict from test data
def cropYieldPredict():
    global model, le_state, le_district, le_season, le_crop
    text.delete('1.0', END)

    testname = askopenfilename(initialdir="Dataset")
    test = pd.read_csv(testname)
    test.fillna(0, inplace=True)

    try:
        test['State_Name'] = le_state.transform(test['State_Name'])
        test['District_Name'] = le_district.transform(test['District_Name'])
        test['Season'] = le_season.transform(test['Season'])
        test['Crop'] = le_crop.transform(test['Crop'])
    except Exception as e:
        text.insert(END, f"❌ Error during label encoding test data:\n{e}\n")
        return

    test_data = normalize(test.values)
    predictions = model.predict(test_data)

    for i, pred in enumerate(predictions):
        production = pred * 100
        crop_yield = (production / 10000) / 10
        text.insert(END, f"Test Record {i + 1}: Production = {production:.2f} KGs\n")
        text.insert(END, f"Yield = {crop_yield:.2f} tones/acre\n\n")

def close():
    main.destroy()

# ================= UI Setup =================
title_font = ('Segoe UI', 20, 'bold')
label_font = ('Segoe UI', 12, 'bold')
button_font = ('Segoe UI', 11, 'bold')
text_font = ('Consolas', 11)

# Title
title = Label(main, text='Crop Yield and Field Prediction',
              bg="#34495e", fg="white", font=title_font, pady=20)
# title.pack(fill=X)
title.pack(fill='x', padx=0, pady=(0, 10))

# Button frame
button_frame = Frame(main, bg="#f0f0f5")
button_frame.place(x=980, y=150)

btn_opts = {
    'bg': '#2ecc71', 'fg': 'white', 'activebackground': '#27ae60',
    'width': 30, 'font': button_font, 'bd': 0, 'pady': 8
}

Button(button_frame, text="Upload Crop Dataset", command=upload, **btn_opts).pack(pady=10)
Button(button_frame, text="Preprocess Dataset", command=processDataset, **btn_opts).pack(pady=10)
Button(button_frame, text="Train ML Model", command=trainModel, **btn_opts).pack(pady=10)
Button(button_frame, text="Upload Test Data & Predict Yield", command=cropYieldPredict, **btn_opts).pack(pady=10)
Button(button_frame, text="Close", command=close,
       bg="#e74c3c", activebackground="#c0392b", fg="white", font=button_font, width=30, bd=0, pady=8).pack(pady=10)

# Text display area
text_frame = Frame(main)
text_frame.place(x=30, y=130)

text = Text(text_frame, height=34, width=110, font=text_font, wrap=WORD, bg="white", fg="#2c3e50", bd=1, relief=SOLID)
scroll = Scrollbar(text_frame, command=text.yview)
text.configure(yscrollcommand=scroll.set)
text.pack(side=LEFT, fill=BOTH)
scroll.pack(side=RIGHT, fill=Y)

# Start the app
main.mainloop()
