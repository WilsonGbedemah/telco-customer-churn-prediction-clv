
# AI Usage Documentation

This document outlines how the AI assistant, Gemini, was used as a pair-programming partner throughout the development of the Telco Customer Churn Prediction project.

## Role of the AI Assistant

Gemini's primary role was to accelerate development by handling code generation, executing commands, and assisting in analysis and debugging, guided by human direction and verification.

## Key Areas of AI Assistance

The AI's involvement can be broken down into several key areas:

### 1. Project Scaffolding and Execution

*   **Command Execution:** Gemini was used to execute all shell commands via the project's `Makefile`. This included setting up the virtual environment (`make install`), running the data pipeline (`make all`), and launching the application (`make app`).
*   **File System Operations:** Gemini was used for all file system interactions, such as writing and updating scripts (`.py` files), creating documentation (`.md` files), and reading file contents for review.

### 2. Code Generation

Gemini generated the complete, first-draft Python scripts for every stage of the project based on the requirements in `overview.txt`. This included:

*   `data_prep.py`: Script for data cleaning, feature engineering, and splitting.
*   `clv_analysis.py`: Script to calculate Customer Lifetime Value and analyze its relationship with churn.
*   `train_models.py`: Script to train, tune, and evaluate the three machine learning models.
*   `interpretability.py`: Script to calculate and save feature importances.
*   `predict.py`: Script containing the function for making on-demand predictions.
*   `app.py`: The final Streamlit application script.

### 3. Debugging and Error Correction

This was one of the most critical areas of assistance. The AI was instrumental in diagnosing and fixing numerous issues that arose during development:

*   **Dependency Conflicts:** Helped troubleshoot and resolve package installation failures, particularly with `scikit-learn` and `shap`, by iteratively adjusting the `requirements.txt` file.
*   **Syntax Errors:** Identified and corrected multiple `SyntaxError` issues in Python scripts, such as unterminated strings and incorrect quoting in `print` statements.
*   **Logic Errors:** Found and fixed a critical bug in the `predict.py` script's `make_prediction` function, where the one-hot encoding logic for single predictions was flawed. The AI rewrote the function to be more robust.
*   **Rendering Errors:** Corrected a subtle indentation issue in the `app.py` script's `st.markdown` component that would have caused incorrect rendering of text.

### 4. Analysis and Model Tuning

The AI's analytical capabilities were guided by specific, high-level prompts from the user.

*   **Prompt:** *"Review the model performance. The recall is below 60%. What is the likely cause and how can we fix it?"*
    *   **AI Action:** This led Gemini to identify the class imbalance problem and propose the solution of using `class_weight='balanced'` and `scale_pos_weight` in the models. This successfully raised the Recall score above the required threshold.

*   **Prompt:** *"Can we tune the model to achieve Precision >= 60% and Recall >= 70%?"*
    *   **AI Action:** This prompted Gemini to suggest and implement Decision Threshold Tuning. While the dual goal was not achievable, the experiment provided a clear understanding of the Precision/Recall trade-off.

*   **Prompt:** *"My goal is now a recall of at least 80%. Which model should I use?"*
    *   **AI Action:** Based on previous results, Gemini identified that the balanced Logistic Regression model already met this goal, leading to the final model selection.

## Verification and Oversight

Throughout the project, the user provided continuous oversight, verifying the AI-generated code, validating the analytical insights, and making all final decisions on the project's direction. This collaborative workflow, with the AI handling execution and first-draft generation and the human providing strategic direction and verification, was key to the project's success.
