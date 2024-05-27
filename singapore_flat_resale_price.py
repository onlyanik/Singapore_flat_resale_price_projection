import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import statistics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import pickle
import streamlit as st

df= pd.read_csv("D:\\Coding\\ResaleFlatPricesBasedonApprovalDate19901999.csv")
# print(df)
df1=pd.read_csv("D:\\Coding\\ResaleFlatPricesBasedonApprovalDate2000Feb2012.csv")
# print(df1)
df2=pd.read_csv("D:\\Coding\\ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv")
# print(df2)
df3=df2=pd.read_csv("D:\\Coding\\ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv")
# print(df3)
df4= df2=pd.read_csv("D:\Coding\\\ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv")

#merging all data into a single one

frame=[df,df1,df2,df3,df4]

result= pd.concat(frame)
# print(result)

#removing null values

result.isnull().sum() 
# print(result1)
result1 = result.dropna()
# result2 = result1.isnull().sum()

# print(result1)
# print(result1[['resale_price', 'floor_area_sqm', 'lease_commence_date']])
print(result1.dtypes)


#splitting store range e.g.10 to 12 and also finding out median

def get_median(x):
    split_list = x.split(' TO ')
    float_list = [float(i) for i in split_list]
    median = statistics.median(float_list)
    return median

df['storey_median'] = result1['storey_range'].apply(lambda x: get_median(x))
df
scope_df = df[['cbd_dist','min_dist_mrt','floor_area_sqm','lease_remain_years','storey_median','resale_price']]
scope_df

scope_df = scope_df.drop_duplicates()
scope_df

#finding skewness of every column

col = ['cbd_dist','min_dist_mrt','floor_area_sqm','lease_remain_years','storey_median','resale_price']

for i in col:
    plt.figure(figsize=(8, 6))  
    sns.boxplot(data=df, x=i)
    plt.title(f'Boxplot of {i}')
    plt.xlabel(i)
    plt.show()

#applying logarthm function to the required column to handle skewness, rest of the column 
df1['floor_area_sqm'] = np.log(df1['floor_area_sqm'])
sns.boxplot(x='floor_area_sqm', data=df1)
plt.show()

df1['storey_median'] = np.log(df1['storey_median'])
sns.boxplot(x='storey_median', data=df1)
plt.show()

df1['resale_price'] = np.log(df1['resale_price'])
sns.boxplot(x='resale_price', data=df1)
plt.show()

df1.dtypes

# visualizing different columns  using correlation matrix
corrMatrix = df1.corr()
plt.figure(figsize=(15, 10))
plt.title("Correlation Heatmap")
sns.heatmap(
    corrMatrix, 
    xticklabels=corrMatrix.columns,
    yticklabels=corrMatrix.columns,
    cmap='RdBu', 
    annot=True
)

#Encoding data & its normalization

X=df1[['cbd_dist','min_dist_mrt','floor_area_sqm','lease_remain_years','storey_median']]
y=df1['resale_price']

scaler = StandardScaler()
X = scaler.fit_transform(X)

#Splitting train, test model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#Decision tree regressor

# Decision Tree Regressor
dtr = DecisionTreeRegressor()

# hyperparameters
param_grid = {
    'max_depth': [2, 5, 10, 15, 20, 22],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
    'max_features': ['auto', 'sqrt', 'log2']
}

# gridsearchcv
grid_search = GridSearchCV(estimator=dtr, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# evalution metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(" ")
print('Mean squared error:', mse)
print('Mean Absolute Error', mae)
print('Root Mean squared error:', rmse)
print(" ")
print('R-squared:', r2)

#Testing traned model
new_sample = np.array([[8740, 999, np.log(44), 55, np.log(11)]])
new_sample = scaler.transform(new_sample[:, :5])
new_pred = best_model.predict(new_sample)[0]
np.exp(new_pred)

# Saving the model

with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)




# Define a function to save the model and scaler
def save_objects(model, scaler):
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    st.success('Model and Scaler saved successfully!')

# Dummy model and scaler (replace these with your actual model and scaler)
best_model = "Your trained model here"
scaler = "Your data scaler here"

# Streamlit interface
st.title("Save Machine Learning Model and Scaler")
st.write("Click the button below to save the model and scaler.")

if st.button('Save Model and Scaler'):
    save_objects(best_model, scaler)



