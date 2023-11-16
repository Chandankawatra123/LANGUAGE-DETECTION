# Multilingual Text Classification: Language Detection with Machine Learning Algorithms


## Description

Our project focuses on the development and comparison of machine learning models for accurate language detection in multilingual text data. Language detection is a crucial component in various applications, such as content filtering, language-specific services, and global customer support. By leveraging different machine learning algorithms, we aim to build a robust language detection system that excels in accuracy, efficiency, and adaptability.

## Code 

IMPORTING LIBRARIES AND DATASET

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


## Requirements

Make sure you have the following dependencies installed:

- Python 
- Scikit-Learn 
- tessract
