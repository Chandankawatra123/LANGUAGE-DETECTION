# Multilingual Text Classification: Language Detection with Machine Learning Algorithms


## Description

Our project focuses on the development and comparison of machine learning models for accurate language detection in multilingual text data. Language detection is a crucial component in various applications, such as content filtering, language-specific services, and global customer support. By leveraging different machine learning algorithms, we aim to build a robust language detection system that excels in accuracy, efficiency, and adaptability.

## Code 

Importing Libraries and Dataset 

![image](https://github.com/Chandankawatra123/LANGUAGE-DETECTION/assets/100226305/1d4c0dea-4b5e-4efe-aa6b-c55ce7d98c7e)

Printing Language Distribution

![image](https://github.com/Chandankawatra123/LANGUAGE-DETECTION/assets/100226305/816520b8-862a-4ed3-87e5-3fa61a88cfb0)

![image](https://github.com/Chandankawatra123/LANGUAGE-DETECTION/assets/100226305/1b408240-a39d-4c72-af83-d80ae23d992c)

Converting categorical variables to numerical

![image](https://github.com/Chandankawatra123/LANGUAGE-DETECTION/assets/100226305/d0a52fa2-4c16-4824-aa0c-1eb743b67517)

Removing Symbols and Numbers

![image](https://github.com/Chandankawatra123/LANGUAGE-DETECTION/assets/100226305/5c28ceec-f435-4345-9de4-95acd3727998)

Train-Test Splitting(80-20)

![image](https://github.com/Chandankawatra123/LANGUAGE-DETECTION/assets/100226305/07e75c0c-98bb-4649-80bf-ff5494280268)

Function For Predicting Language 

![image](https://github.com/Chandankawatra123/LANGUAGE-DETECTION/assets/100226305/46c4688c-bc83-4acb-9c4d-d85eb79e40ec)

Extracting Data from Image and calling predict if text exists

![image](https://github.com/Chandankawatra123/LANGUAGE-DETECTION/assets/100226305/1612d862-f8a7-46b7-82ca-504cc46adc6b)


## Model-Creation and Prediction -- 1 (MultinomialNB)

![image](https://github.com/Chandankawatra123/LANGUAGE-DETECTION/assets/100226305/f3656aa1-6bfd-4574-a551-c6b4e6fd0373)

Accuracy , Confusion Matrix and Classification report

![image](https://github.com/Chandankawatra123/LANGUAGE-DETECTION/assets/100226305/d720f31a-a29d-49b5-969b-00f125120c4f)

![image](https://github.com/Chandankawatra123/LANGUAGE-DETECTION/assets/100226305/6c1a0c9b-da1b-4f0d-aa8a-d4686b8da658)

## Model-Creation and Prediction -- 2 (Random Forest)

Accuracy , Confusion Matrix and Classification report

![image](https://github.com/Chandankawatra123/LANGUAGE-DETECTION/assets/100226305/576b7145-1c9c-4fcb-9206-6f63474d3560)
![image](https://github.com/Chandankawatra123/LANGUAGE-DETECTION/assets/100226305/91f40627-4960-4cd9-b38a-34bd667a1353)

## Model-Creation and Prediction -- 3 (K Nearest Neighbour)

Accuracy , Confusion Matrix and Classification report

![image](https://github.com/Chandankawatra123/LANGUAGE-DETECTION/assets/100226305/19277e65-2610-40be-aef0-31afc0612b4e)
![image](https://github.com/Chandankawatra123/LANGUAGE-DETECTION/assets/100226305/4a84e259-a2b3-450a-be64-ee0f452a7155)


## NAIVE BAYES 
Strengths:
Efficient and performs well on high-dimensional data.
Particularly effective for text classification and spam filtering.

## RANDOM FOREST
Strengths:
Robust and handles non-linear relationships well.
Can capture complex interactions between features.


##WHY NAIVE BAYES AND RANDOM FOREST OVER KNN

Use Naive Bayes When:

Dealing with text data, especially in natural language processing (NLP) tasks.
Features are conditionally independent given the class label.
Computational efficiency is crucial, as Naive Bayes is computationally inexpensive.
Use Random Forest When:

The dataset exhibits complex relationships and interactions between features.
Robustness against overfitting is essential.
A balance between interpretability and predictive performance is needed.




## Requirements

Make sure you have the following dependencies installed:

- Python 
- Scikit-Learn 
- tessract
