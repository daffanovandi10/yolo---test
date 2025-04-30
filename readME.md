# Smart Scheduling for Predictive Maintenance

## Overview

This project details the development of a predictive maintenance model aimed at forecasting machine failures and informing optimized maintenance schedules. By analyzing sensor data, the goal is to proactively identify potential issues, thereby minimizing operational downtime and maintenance expenditures. The complete analytical workflow, from initial data exploration to final model evaluation, is documented within this Jupyter Notebook.

## Project Goals

* To predict the likelihood of machine failure using sensor readings.
* To identify the key sensor features that are most indicative of potential failures.
* To evaluate the performance and reliability of the developed predictive model using cross-validation.
* To explore the potential for cost optimization in maintenance scheduling based on model predictions.

## Data Description

* **Data Source:** The dataset utilized in this project appears to be synthetically generated sensor data, simulating the operational behavior of a machine under various conditions.
* **Data Format:** The data is structured and manipulated using the Pandas DataFrame in Python.
* **Key Features (Sensor Readings):**
    * `temperature`: Represents the machine's operating temperature.
    * `pressure`: Indicates the machine's operating pressure levels.
    * `vibration`: Measures the machine's vibration intensity.
    * `current`: Represents the electrical current drawn by the machine.
    * `humidity`: Indicates the ambient humidity levels.
* **Target Variable:** `failure` - A binary outcome variable where 1 signifies a machine failure within a specific prediction window, and 0 indicates no failure.
* **Temporal Aspect:** While not a strict time series analysis, the dataset likely contains snapshots of sensor readings taken over time, capturing conditions leading up to potential failures.

## Jupyter Notebook: `Smart Scheduling Predictive Maintenance.ipynb`

The Jupyter Notebook systematically guides through the process of building and evaluating the predictive maintenance model, including hyperparameter tuning using cross-validation:

1.  **Data Loading and Initial Exploration:**
    * The analysis begins by loading the dataset into a Pandas DataFrame.
    * Initial steps involve examining the dataset's structure, data types, and generating descriptive statistics to gain a preliminary understanding of the data.
    * The presence of missing values is checked, and in this case, they are handled by imputing with the mean of the respective columns.
    * The distribution of the `failure` target variable is assessed to understand the balance between failure and non-failure instances.

2.  **Exploratory Data Analysis (EDA):**
    * **Sensor Distribution Analysis:** Histograms and box plots are generated to visualize the distribution of each sensor reading (`temperature`, `pressure`, `vibration`, `current`, `humidity`) separately for instances where a failure occurred and where it did not. This comparative visualization helps identify potential differences in sensor behavior leading up to failures. For instance, we can observe if failures are more frequent at extreme values of certain sensors.
    * **Correlation Analysis:** A correlation matrix is computed and visualized as a heatmap to explore the linear relationships between the different sensor readings and their correlation with the `failure` target variable. This step helps identify features that might be strongly associated with the target or exhibit multicollinearity.

3.  **Feature Engineering:**
    * A new feature, `vibration_pressure_ratio`, is created by combining the `vibration` and `pressure` readings. This demonstrates an attempt to capture potential interaction effects between different sensor measurements that might be indicative of impending failures.
    * Numerical sensor features are scaled using `StandardScaler`. This standardization ensures that all features contribute equally during model training, preventing features with larger magnitudes from dominating the learning process.

4.  **Model Selection, Training, and Hyperparameter Tuning:**
    * The dataset is split into training and testing subsets to allow for an unbiased evaluation of the model's ability to generalize to unseen data.
    * A classification model (the specific type is detailed in the notebook) is selected.
    * **Hyperparameter Tuning with Cross-Validation:** The notebook utilizes cross-validation (`Fitting 5 folds for each of 8 candidates, totalling 40 fits`) to find the optimal hyperparameters for the chosen model. This process involves training and evaluating the model multiple times on different subsets of the training data to maximize its performance.

5.  **Model Evaluation:**
    * The best model found through cross-validation is then evaluated on the held-out test dataset.
    * **Confusion Matrix:** The confusion matrix for the test set is:
        ```
        [[1078  612]
         [ 245 1379]]
        ```
        This shows:
        * True Negatives (TN): 1078 (correctly predicted no failure)
        * False Positives (FP): 612 (incorrectly predicted failure when there was none)
        * False Negatives (FN): 245 (incorrectly predicted no failure when there was a failure)
        * True Positives (TP): 1379 (correctly predicted failure)
    * **Classification Report:** The classification report provides precision, recall, F1-score, and support for each class (0: no failure, 1: failure):
        ```
              precision    recall  f1-score   support

           0       0.81      0.64      0.72      1690
           1       0.69      0.85      0.76      1624

        accuracy                           0.74      3314
       macro avg       0.75      0.74      0.74      3314
    weighted avg       0.75      0.74      0.74      3314
        ```
        * **Class 0 (No Failure):** Precision is 0.81 (when the model predicted no failure, it was correct 81% of the time), Recall is 0.64 (the model identified 64% of all actual no-failure cases), and the F1-score is 0.72.
        * **Class 1 (Failure):** Precision is 0.69 (when the model predicted a failure, it was correct 69% of the time), Recall is 0.85 (the model identified 85% of all actual failure cases), and the F1-score is 0.76.
        * **Accuracy:** The overall accuracy of the model on the test set is 0.74 (74%).
    * **ROC-AUC Score:** The Area Under the Receiver Operating Characteristic curve is 0.8471. This indicates a good ability of the model to distinguish between failure and non-failure cases.

## Key Findings and Observations

* **Cross-Validation for Robustness:** The use of 5-fold cross-validation ensures a more robust evaluation of the model's performance by assessing it on multiple independent subsets of the training data.
* **Confusion Matrix Insights:** The confusion matrix reveals a notable number of false positives and false negatives, indicating areas where the model's predictions could be improved. There's a tendency to predict more false positives than false negatives in this evaluation.
* **Balanced Performance:** The classification report shows a reasonable balance between precision and recall for both classes, although there's a trade-off. The model is better at identifying actual failures (higher recall for class 1) but has more false alarms (lower precision for class 1). Conversely, it's more conservative in predicting no failure (higher precision for class 0) but misses more actual no-failure cases (lower recall for class 0).
* **Good Discriminatory Power:** The ROC-AUC score of 0.8471 suggests that the model has a good ability to discriminate between instances that will lead to failure and those that will not.

## Potential Next Steps

* Further investigate the reasons for the relatively high number of false positives and false negatives. This might involve more detailed error analysis or exploring different model architectures.
* Consider adjusting the classification threshold to optimize for specific business needs (e.g., minimizing false negatives even if it increases false positives in scenarios where missing a failure is very costly).
* Explore more advanced feature engineering techniques or consider incorporating additional data sources to potentially improve the model's predictive accuracy.
* Evaluate the economic implications of the current model's performance, considering the costs of false positives (unnecessary maintenance) and false negatives (unpredicted failures).

## Installation

The analysis in this notebook relies on the following Python libraries:

pandas numpy matplotlib seaborn scikit-learn

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
Author
Muhammad Daffa Novandi
rmdaffanovandi@gmail.com