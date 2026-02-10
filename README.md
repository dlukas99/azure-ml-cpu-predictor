# CPU Base Clock Prediction Service

An end-to-end Machine Learning solution designed to predict processor base clock speeds based on hardware specifications. This project leverages cloud-based inference for real-time predictions and provides a comprehensive data analytics dashboard built with Streamlit.

## 🛠 Technology Stack

- **Cloud Platform**: Azure Machine Learning
- **Primary Algorithm**: Optimized Random Forest Regressor ($R^2 = 0.90$)
- **Experiment Tracking**: MLflow
- **API Deployment**: Azure Container Instance (ACI)
- **Frontend**: Streamlit (Python)

## 📁 Project Structure

- `app.py` – Streamlit web application providing a user interface for single and batch predictions.
- `cpu.csv` – Dataset containing processor specifications used for training and testing.
- `cpu_train.ipynb` – Jupyter notebook containing the full pipeline: EDA, feature selection, and Azure ML training logic.
- `conda_env.yml` – Configuration file defining the environment and dependencies (Python 3.9, Scikit-learn, etc.).
- `score.py` – Inference script used by the Azure endpoint to process incoming JSON requests.
- `requirements.txt` – List of Python dependencies required for the client application.

## 📊 Methodology & Model Comparison

During the development phase, five different regression algorithms were evaluated to find the most accurate predictor:

| Algorithm | R2 Score |
|---|---|
| Optimized Random Forest | 0.90 |
| Gradient Boosting | 0.88 |
| Decision Tree | 0.78 |
| Linear Regression | 0.54 |
| ElasticNet | 0.52 |

The Random Forest model was optimized using GridSearchCV to tune hyperparameters such as tree depth and the number of estimators, achieving the highest precision.

## 🔄 Data Engineering & Feature Selection

The project implements a rigorous data pipeline to ensure high model generalization:

- **EDA (Exploratory Data Analysis)**: Statistical profiling using boxplots and correlation heatmaps to identify key drivers of clock speed.
- **Data Cleaning**: Handling missing values with median imputation and removing redundant descriptive columns.
- **Advanced Feature Selection**: Used the SelectKBest algorithm with f_regression to isolate the 5 most statistically significant features: Cores, Threads, Boost Clock, Process technology, and TDP.
