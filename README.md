# Loan_Prediction_for_marketing_campaign_of_bank
Machine Learning Model which can predict if a Customer would take up Personal Loan if the targeted Marketing Campaign is done.

Overview
This project focuses on predicting whether customers of a bank would accept a personal loan offer during a targeted marketing campaign. The bank aims to increase the success rate of converting liability customers into personal loan customers while retaining them as depositors. Historical data from a previous campaign is used to build a machine learning model for prediction.

Use Case
The bank's previous campaign targeted liability customers and achieved a conversion rate of over 9%. This success has prompted the retail marketing department to devise more targeted campaigns with improved success ratios using minimal budget.

Objective
Build a machine learning model using the dataset provided to predict if a customer would accept a personal loan offer during a targeted marketing campaign.

Dataset Description
The dataset consists of the following variables:

1. ID: Customer ID
2. Age: Customer's age in years
3. Experience: Number of years of professional experience
4. Income: Annual income in thousands of dollars
5. Postal Code: Postal code of the customer's home address
6. Family Size: Number of family members
7. CCAvgSpending: Average credit card spending per month in thousands of dollars
8. Education: Customer's education level (Undergrad, Graduate, Advanced Degree)
9. Mortgage: Value of home mortgage in thousands of dollars (if any)
10. Investment Account: Whether the customer has an investment account with the bank (1 = Yes, 0 = No)
11. Deposit Account: Whether the customer has a deposit account with the bank (1 = Yes, 0 = No)
12. InternetBanking: Whether the customer uses internet banking (Yes, No)
13. Personal Loan: Whether the customer accepted the personal loan offered in the last campaign (Yes, No)

Files Included
1. Loan_Prediction.ipynb - Jupyter notebook containing data exploration, preprocessing, model training, evaluation, and prediction code.

2. Loan_Prediction_dataset.xlsx - Excel file containing the dataset used for training and testing the model.

3. requirements.txt - List of Python packages and versions required to set up the environment.

4. utils_Loan_Prediction.py - Common utility functions used for different tasks in cleaning, transformations and plotting of the data

Steps to Run the Project
1. Clone the repository:
   git clone <repository-url>
    cd <repository-name>

2. Install the required Python packages:
   pip install -r requirements.txt

3. Open and execute the Jupyter notebook Loan_Prediction.ipynb to explore the data, train the machine learning model, and predict customer responses.



