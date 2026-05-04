import pandas as pd 
import numpy as np 
import joblib
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, precision_score, recall_score,f1_score)
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway, t

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("X_Test", X_test)
    print("X_train", X_train)


    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    print("Model training complete.")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    evaluate_model(y_test, y_pred)

    # Save the model
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / 'random_forest_model.pkl')
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the trained model to a file
    model_filename = models_dir / 'random_forest_model.pkl'
    joblib.dump(model, model_filename)
    print(f"Trained model saved to {model_filename}")

    return model


def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))



def forecast_clean_energy_production(years_ahead=5):
    # Load historical energy data
    data_path = Path(__file__).resolve().parent.parent / "Data" / "States_Annual_Energy_Generation_Sources_1990_2019.xlsx"
    df = pd.read_excel(data_path, sheet_name='Net_Generation_1990-2019 Final', header=3)
    df.columns = ['YEAR', 'STATE', 'TYPE OF PRODUCER', 'ENERGY SOURCE', 'GENERATION (Megawatthours)']
    
    # explain about the renewable resources 
    renewable_sources = ['Hydroelectric Conventional', 'Wind', 'Solar Thermal and Photovoltaic', 'Wood and Wood Derived Fuels', 'Other Biomass', 'Geothermal', 'Other Gases']
    df_renewable = df[df['ENERGY SOURCE'].isin(renewable_sources)]
    yearly_renewable = df_renewable.groupby('YEAR')['GENERATION (Megawatthours)'].sum()
    
    # take a arima model here 
    model = auto_arima(yearly_renewable, seasonal=False, suppress_warnings=False)
    forecast = model.predict(n_periods=years_ahead)
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / 'clean_energy_forecast_model.pkl')
    
    # Plot prediction here 
    plot_forecast(yearly_renewable, forecast, years_ahead)
    
    return forecast, model


def plot_forecast(historical, forecast, years_ahead):
    output_dir = Path(__file__).resolve().parent.parent / "Output"
    output_dir.mkdir(exist_ok=True)  
    plt.figure(figsize=(12, 6))
    plt.plot(historical.index, historical.values, label='Historical Data', color='blue')
    forecast_years = range(historical.index.max() + 1, historical.index.max() + 1 + years_ahead)
    plt.plot(forecast_years, forecast, label='Forecast', color='red', linestyle='--')
    plt.title('Clean Energy Production Forecast')
    plt.xlabel('Year')
    plt.ylabel('Generation (MWh)')
    plt.legend()
    plt.savefig(output_dir / 'clean_energy_forecast.png')
    plt.close()
    print(f"Forecast plot saved to {output_dir / 'clean_energy_forecast.png'}")


def forecast_ev_adoption():
    data_path = Path(__file__).resolve().parent.parent / "Processed_Data" / "clean_data.csv"
    df = pd.read_csv(data_path)
    X = df[['total_generation']]
    y = df['ev_count']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # evalute the model performance 
    evaluate_forecasting_model(y_test, y_pred)
    
    # Plot predictions
    plot_ev_predictions(y_test, y_pred)
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / 'ev_adoption_model.pkl')
    return model


def plot_ev_predictions(y_test, y_pred):
    output_dir = Path(__file__).resolve().parent.parent / "Output"
    output_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual EV Count')
    plt.ylabel('Predicted EV Count')
    plt.title('EV Adoption Model: Actual vs Predicted')
    plt.savefig(output_dir / 'ev_adoption_predictions.png')
    plt.close()
    print(f"EV predictions plot saved to {output_dir / 'ev_adoption_predictions.png'}")


def evaluate_forecasting_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}




def policy_impact_assessment():
    print("Policy Impact Assessment:")
    print("Increasing clean energy production can support higher EV adoption.")
    print("Policies promoting renewable energy may accelerate EV market penetration.")
    print("Forecasting models can help policymakers plan infrastructure investments.")
    
    try:
        models_dir = Path(__file__).resolve().parent.parent / "models"
        ev_model = joblib.load(models_dir / 'ev_adoption_model.pkl')
        energy_model = joblib.load(models_dir / 'clean_energy_forecast_model.pkl')
        sample_generation = np.array([[200000000]])  
        predicted_ev = ev_model.predict(sample_generation)
        print(f"Predicted EV count for increased generation: {predicted_ev[0]:.0f}")
        forecast_values = energy_model.predict(n_periods=3)
        print(f"Forecasted clean energy for next 3 years: {forecast_values.values}")
        
    except FileNotFoundError:
        print("Models not found. Please train models first.")



def hypothesis_testing():     
    # Statistical Tests
    data_path = Path(__file__).resolve().parent.parent / "Processed_Data" / "clean_data.csv"
    df = pd.read_csv(data_path)
    print("df", df)
    median_gen = df['total_generation'].median()
    high_gen_ev = df[df['total_generation'] > median_gen]['ev_count']
    print("high_gen_ev", high_gen_ev)
    low_gen_ev = df[df['total_generation'] <= median_gen]['ev_count']
    print("low_gen_ev", low_gen_ev)
    
    # Hypothesis Definition
    print("\nHypothesis Testing:")
    print("Null Hypothesis (H0): There is no significant difference in EV adoption between high and low energy generation states.")
    print("Alternative Hypothesis (H1): There is a significant difference in EV adoption between high and low energy generation states.")
    
    # T-test for means (independent samples)
    t_val, p_val = ttest_ind(high_gen_ev, low_gen_ev)
    print("t_val", t_val)
    print("p_val", p_val)
    
    alpha = 0.05
    df_degrees = len(high_gen_ev) + len(low_gen_ev) - 2  
    print("df_degrees", df_degrees)
    
    c_t = t.ppf(1 - alpha/2, df_degrees)
    print("c_t", c_t)
    
    print(f"\nT-Test Results (comparing means): T-statistic = {t_val:.2f}, P-value = {p_val:.4f}")
    print(f"Critical t-value: {c_t:.2f}")
    
    print('T-test:')
    if np.abs(t_val) > c_t:
        print('Significant difference found.')
    else:
        print('No significant difference.')
    
    print('P-test:')
    if p_val < alpha:
        print('Reject H0 (significant difference)')
    else:
        print('Fail to reject H0 (no significant difference)')

    
#if __name__ == "__main__":
#    policy_impact_assessment()
#    print("\n")
#    analyze_incentive_effectiveness()
#    print("\n")
#    conclusion_and_recommendations()
#    print("\n")
#    print("t_test:", t_val)
#    print("p_val:", p_val)
#    print("df_degrees:", df_degrees)
#    print("c_t:", c_t)




def predict_ev_high_low(new_generation):
    #Predict if EV adoption is high or low based on total generation using Random Forest model.
    models_dir = Path(__file__).resolve().parent.parent / "models"
    model = joblib.load(models_dir / 'random_forest_model.pkl')
    prediction = model.predict([[new_generation]])
    return "High EV adoption" if prediction[0] == 1 else "Low EV adoption"


def predict_ev_count(new_generation):
    #Predict EV count based on total generation using Linear Regression model.
    models_dir = Path(__file__).resolve().parent.parent / "models"
    model = joblib.load(models_dir / 'ev_adoption_model.pkl')
    prediction = model.predict([[new_generation]])
    return prediction[0]


def plot_predictions():
   # Plot predictions for EV count and adoption level across a range of generation values.
    models_dir = Path(__file__).resolve().parent.parent / "models"
    rf_model = joblib.load(models_dir / 'random_forest_model.pkl')
    lr_model = joblib.load(models_dir / 'ev_adoption_model.pkl')
    
    # Generate range of generation values
    generation_range = np.linspace(0, 500000000, 100)
    
    # Predict EV count
    ev_counts = lr_model.predict(generation_range.reshape(-1, 1))
    
    # Predict high/low
    high_low = rf_model.predict(generation_range.reshape(-1, 1))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # EV count plot
    ax1.plot(generation_range, ev_counts, color='blue', label='Predicted EV Count')
    ax1.set_title('Predicted EV Count vs Total Generation')
    ax1.set_xlabel('Total Generation (MWh)')
    ax1.set_ylabel('EV Count')
    ax1.legend()
    
    # High/Low plot
    colors = ['red' if x == 0 else 'green' for x in high_low]
    ax2.scatter(generation_range, high_low, c=colors, alpha=0.7)
    ax2.set_title('EV Adoption Level vs Total Generation')
    ax2.set_xlabel('Total Generation (MWh)')
    ax2.set_ylabel('Adoption Level')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Low', 'High'])
    
    plt.tight_layout()
    
    output_dir = Path(__file__).resolve().parent.parent / "Output"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'predictions_plot.png')
    plt.close()
    print(f"Predictions plot saved to {output_dir / 'predictions_plot.png'}")


if __name__ == "__main__":
    #load a data for classification model
    data_path = Path(__file__).resolve().parent.parent / "Processed_Data" / "clean_data.csv"
    df = pd.read_csv(data_path)
    median_ev = df['ev_count'].median()
    print("median_ev", median_ev)
    print("df", df.head())


    y = (df['ev_count'] > median_ev).astype(int)
    X = df[['total_generation']]
    train_model(X, y)
    print("Model training complete.", train_model(X, y))

    forecast_energy, energy_model = forecast_clean_energy_production(years_ahead=5)
    print("forcasted clean energy :", forecast_energy)
    for i, value in enumerate(forecast_energy, 1):
        print(f"Year {2020 + i}: {value:,.0f}")  
    ev_model = forecast_ev_adoption()
    print("ev model :", ev_model)
    policy_impact_assessment()
    print("Starting Hypothesis Testing:", hypothesis_testing())
    hypothesis_testing()
    print("testing is  done:")

    plot_predictions()
    print("All plots are generated ")
    
    












