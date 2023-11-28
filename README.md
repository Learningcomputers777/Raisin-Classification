# Raisin-Classification by Image Input
This is used to classify the raisins of two types “Kecimen” and “Besni” with a interface and deployable application using Flask.  

A machine learning model is made with Support Vector Machine(SVM). The model is trained on the dataset which has 900 pieces data of raisins in which it contains many parameters/features . 

450 are of “Kecimen” and 450 are of “Besni”.  

In the model Kecimen and Besni are referred to as [0] and [1]. 
# Steps to deploy 
### 1. Clone the Repo
### 2. Environment Setup
```python
python3 -m venv raisinclassifyenv
```
```python
source raisinclassifyenv/bin/activate
```
```python
pip install -r requirements.txt
```
### 3. Run Flask App
```python
python app.py 
```

Dataset:https://www.muratkoklu.com/datasets/