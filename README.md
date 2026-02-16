# Student Performance Analysis & Prediction

A Machine Learning project focused on analyzing how various factors (demographic and academic) influence student test scores. I built a predictive model to estimate a student's overall performance using Linear Regression.

##  Project Overview
In this project, I moved beyond looking at individual subject scores by performing **Feature Engineering**. I calculated an **Average Score** from three subjects (Math, Reading, and Writing) to create a single, comprehensive target variable. This allows for a more holistic prediction of student success.

##  Project Structure
The repository is organized as follows:
* **data/**: Contains the raw dataset and the cleaned/engineered version (student_prediction_cleaned.csv).
* **models/**: Contains the serialized Linear Regression model `LR_Model.pkl` and the `StandardScaler.pkl`.
* **notebooks/**: Contains the Jupyter Notebook with the full data analysis and model training steps.

##  Technical Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib/Seaborn
* **Model:** Linear Regression

##  Model Performance
After training and evaluating the model, I achieved the following results:
- **R² Score:** 0.342 (indicating the proportion of variance explained by the model)
- **Mean Absolute Error (MAE):** 8.96



##  Key Insights
* **Feature Engineering:** Creating the `Average_Score` provided a more stable metric for performance than any single subject.
* **Data Preprocessing:** Standardizing the features was a critical step to ensure the model correctly interpreted the weight of each input.
* **Analysis:** The model highlights how preparation and demographic factors contribute to academic outcomes.

##  How to Use
To use the trained model for predictions, you can load the `.pkl` files using `joblib`:
```python
import joblib

model = joblib.load('models/LR_Model.pkl')
scaler = joblib.load('models/StandardScaler.pkl')
# Prepare your data and use model.predict()
