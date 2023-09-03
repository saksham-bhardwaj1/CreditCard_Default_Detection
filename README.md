# Credit-Card-Default-Detection
## Project Video:-
[![Project Video](https://img.youtube.com/vi/moZXgEmJ9BI/maxresdefault.jpg)](https://youtu.be/moZXgEmJ9BI)
#### Internship for PWSkills
### 📝 Overview
This is a classification model for a most common dataset, Credit Card defaulter prediction. Prediction of the next month credit card defaulter based on demographic and last six months behavioral data of customers.

### 🎯 Motivation
There are times when even a seemingly manageable debt, such as credit cards, goes out of control. Loss of job, medical crisis or business failure are some of the reasons that can impact your finances. In fact, credit card debts are usually the first to get out of hand in such situations due to hefty finance charges (compounded on daily balances) and other penalties. A lot of us would be able to relate to this scenario. We may have missed credit card payments once or twice because of forgotten due dates or cash flow issues. But what happens when this continues for months? How to predict if a customer will be defaulter in next months?To reduce the risk of Banks, this model has been developed to predict customer defaulter based on demographic data like gender, age, marital status and behavioral data like last payments, past transactions etc.

### 📈 Dataset Information
In the dataset we have 25 columns which reflect various attributes of the customer. The target column is default.payment.next.month which tells whether a person will default or not. In this dataset, we have information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card holders from April 2005 to September 2005.

Datasource Link: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

### **Technologies Used:**
- Front-End: HTML, CSS
- Back-End: Python (Flask framework)
- MachineLearning: Logitic Regression, DecisionTree , GaussianNB,SupportVector Classifier, GradientBoostingClassifier

### ⚙️ SetUp
### Step 1: Clone the repository
git clone https://github.com/saksham-bhardwaj1/CreditCard_Default_Detection

### Step 2- Create a conda environment after opening the repository
conda create -p venv python==3.8
conda activate venv/

### Step 3 - Install the requirements
pip install -r requirements.txt

### Step 4 - Run the application server
python application.py

### Step 5-
1. Visit the web app. :- http://127.0.0.1:5000/
2. Enter the attributes of the Credit Card in the input form.
3. Click the "Predict" button.
4. Receive the prediction Whether Customer Default on Next month or not.

## **Contributions:**
Contributions to this project are welcome! If you have ideas for improvement, bug fixes, or additional features, feel free to create a pull request or open an issue.
