import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pytesseract
import os

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

warnings.simplefilter("ignore")

# Loading the dataset
data = pd.read_csv(r"C:\Users\91896\Desktop\ML-Language_Detection\MAIN\Language Detection55.csv")

# Display value counts for each language
print("Language Distribution:")
print(data["Language"].value_counts())

# Separating the independent and dependent features
X = data["Text"]
y = data["Language"]

# Converting categorical variables to numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Creating a list for appending the preprocessed text
data_list = []

# Iterating through all the text
for text in X:
    # Removing symbols and numbers
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    # Converting the text to lowercase
    text = text.lower()
    # Appending to data_list
    data_list.append(text)

# Creating bag of words using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()

# Train-test splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Model creation and prediction
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Model evaluation
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)

# Function for predicting language
def predict(text):
    x = cv.transform([text]).toarray()
    lang = model.predict(x)
    lang = le.inverse_transform(lang)
    print("The language is in", lang[0])
    print("ACCURACY: ", ac)

# OCR part
img_path = r'C:\Users\91896\Desktop\ML-Language_Detection\Test Images\MIX\MIX.jpeg'

try:
    osd = pytesseract.image_to_osd(img_path)
    script = re.search("Script: ([a-zA-Z]+)\n", osd).group(1)
    
    if script == "Devanagari":
        text = pytesseract.image_to_string(img_path, lang='hin')
    elif script == "Latin":
        text = pytesseract.image_to_string(img_path)
    
    print("Extracted Text:", text)
    
    if text:
        predict(text)
    else:
        print("No text extracted from the image.")

except Exception as e:
    print("Error in Processing Image:", str(e))
    import traceback
    traceback.print_exc()
