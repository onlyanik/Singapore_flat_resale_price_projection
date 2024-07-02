import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import streamlit as st

# Function to load data
def load_data():
    files = [
        "D:\\Coding\\ResaleFlatPricesBasedonApprovalDate19901999.csv",
        "D:\\Coding\\ResaleFlatPricesBasedonApprovalDate2000Feb2012.csv",
        "D:\\Coding\\ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv",
        "D:\\Coding\\ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv",
        "D:\\Coding\\ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv"
    ]
    dfs = [pd.read_csv(file) for file in files]
    result = pd.concat(dfs, ignore_index=True)
    return result

# Function for feature engineering
def feature_engineering(df):
    df = df.dropna()
    df['storey_median'] = df['storey_range'].apply(lambda x: get_median(x))
    scope_df = df[['floor_area_sqm', 'storey_median', 'resale_price']]
    scope_df = scope_df.drop_duplicates()
    
    # Handling outliers in 'resale_price'
    Q1 = scope_df['resale_price'].quantile(0.25)
    Q3 = scope_df['resale_price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    scope_df = scope_df[(scope_df['resale_price'] >= lower_bound) & (scope_df['resale_price'] <= upper_bound)]
    
    return scope_df

# Function to get median from storey range
def get_median(x):
    split_list = x.split(' TO ')
    float_list = [float(i) for i in split_list]
    return statistics.median(float_list)

# Function to split data
def split_data(df):
    X = df.drop(['resale_price'], axis=1)
    y = df['resale_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

# Function to train and evaluate models
def train_and_evaluate(X_train, y_train, X_test, y_test):
    models = {
        'DecisionTree': DecisionTreeRegressor(),
        'RandomForest': RandomForestRegressor(),
        'GradientBoosting': GradientBoostingRegressor()
    }
    best_model = None
    best_score = -np.inf
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'name': name,
            'model': model,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })
        
        if r2 > best_score:
            best_score = r2
            best_model = model

    for result in results:
        st.write(f"{result['name']} Model Metrics:")
        st.write(f"Mean Squared Error: {result['mse']:.2f}")
        st.write(f"Mean Absolute Error: {result['mae']:.2f}")
        st.write(f"Root Mean Squared Error: {result['rmse']:.2f}")
        st.write(f"R-squared: {result['r2']}")
        st.write("---")
        
    st.write(f'Best Model: {type(best_model).__name__} with R-squared: {best_score:.2f}')
    
    return best_model, results

# Main function to run the app
def main():
    st.title("Singapore Flat Resale Price Prediction")
    
    df = load_data()
    st.write("Data Loaded Successfully!")
    
    scope_df = feature_engineering(df)
    st.write("Feature Engineering Completed!")
    st.write(scope_df.head())

    X_train, X_test, y_train, y_test, scaler = split_data(scope_df)
    st.write("Data Split into Train and Test Sets!")
    
    best_model, results = train_and_evaluate(X_train, y_train, X_test, y_test)
    st.write("Model Training and Evaluation Completed!")

    # Save the best model
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    st.write("Model Saved Successfully!")
    

if __name__ == "__main__":
    main()
