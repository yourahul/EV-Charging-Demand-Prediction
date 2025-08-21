#🔌 EV Vehicle Charging Demand Prediction

This repository contains the end-to-end machine learning project developed as part of my internship at Shell India x AICTE x Edunet Foundation.
The objective is to forecast electric vehicle (EV) adoption trends across counties in Washington State using historical registration data.

The predictions aim to support EV infrastructure planning — particularly for anticipating charging station demand.

🎯 Project Objectives

Analyze EV growth patterns using real-world vehicle registration data

Clean and preprocess the dataset for modeling

Engineer lag-based time-series features for forecasting

Train a Random Forest Regressor model

Forecast EV adoption for the next 36 months

Build a Streamlit-based interactive web app for visualization

Enable county-level EV adoption comparisons and growth insights

🧰 Tools & Technologies Used

Language: Python

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels

Model: RandomForestRegressor

App Framework: Streamlit

IDE: Google Colab + VS Code

Version Control: Git + GitHub

📁 Project Structure
EV_Vehicle_Charge_Demand/
├── EV_Vehicle_Charging_Demand_Prediction.ipynb   # Jupyter Notebook
├── app.py                                       # Streamlit App
├── forecasting_ev_model.pkl                     # Trained Model
├── preprocessed_ev_data.csv                     # Cleaned Dataset
├── ev-car-factory.jpg                           # UI Image

📈 How It Works

Data grouped by county and model year

Feature engineering: lag1, rolling mean, percent change, growth slope

Random Forest model trained on historical EV totals

Forecast horizon: 36 months into the future

Streamlit app allows selecting counties and viewing EV adoption projections

🚀 Future Enhancements

Experiment with advanced time-series models (Prophet, XGBoost, LSTM)

Deploy the Streamlit app on cloud (Streamlit Cloud / Heroku)

Extend dataset beyond Washington State for larger adoption insights
