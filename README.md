# Telco Customer Churn Prediction & CLV Analysis

This project provides an end-to-end solution for predicting customer churn and analyzing customer lifetime value (CLV) for a fictional telecommunications company. The final output is an interactive Streamlit web application that allows users to get on-demand churn predictions, understand the key drivers of churn, and view analysis on customer value.

## Business Problem

Customer churn is a critical issue for subscription-based businesses, leading to significant revenue loss. This project addresses two key business questions:
1.  **Who is likely to churn?** By building a predictive machine learning model.
2.  **Which customers are most valuable to retain?** By calculating and analyzing Customer Lifetime Value.

The goal is to empower the business to proactively identify high-value customers who are at risk of churning and to target them with effective retention strategies.

## Solution Overview

The solution is an interactive Streamlit application with three main functionalities:

*   **Churn Prediction:** Get a real-time churn probability for any customer profile using our trained machine learning model.
*   **Model Performance:** Review the performance metrics (Precision, Recall, AUC-ROC) of the different models and explore the key features that influence their predictions.
*   **CLV Overview:** View an analysis of how customer value relates to churn rate, providing insights into which customer segments are most critical to the business.

## CLV Assumptions

To calculate Customer Lifetime Value, we made a key simplifying assumption:

*   **`ExpectedTenure`**: This was calculated as the average tenure of all non-churning customers in our training dataset. This provides a stable estimate of how long a loyal customer is expected to stay with the company.

## Final Model Performance

The final, recommended model is a **Logistic Regression** classifier, which was optimized to maximize **Recall**—our primary business metric for ensuring we identify the highest number of at-risk customers.

The model was tuned to a specific decision threshold to meet the business requirements, achieving the following final performance on the unseen test data:

*   **Optimal Decision Threshold:** 0.580
*   **Precision:** 54.8%
*   **Recall:** 74.9%
*   **AUC-ROC:** 83.7%

## How to Run the Project

To set up and run this project on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd telco-customer-churn-prediction-clv
    ```

2.  **Create Environment and Install Dependencies:**
    This project uses `make` to streamline setup. The following command will create a virtual environment and install all required packages from `requirements.txt`.
    ```bash
    make install
    ```

3.  **Run the Data and Modeling Pipeline:**
    The `Makefile` includes a command to run all the necessary scripts in order, from data preparation to model training.
    ```bash
    make all
    ```
    This command will prepare the data, run the CLV analysis, train the models, and run the interpretability script.

4.  **Launch the Streamlit Application:**
    Once the pipeline has been run, you can launch the interactive web application.
    ```bash
    make app
    ```
    This will start the Streamlit server. You can then view the application in your web browser at the local URL provided in your terminal (usually `http://localhost:8501`).