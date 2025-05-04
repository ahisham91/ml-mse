import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import matplotlib.pyplot as plt
 
#Q1:
PATH = r'C:\Users\anish\Downloads'
fname1 = 'Data-Covid002.csv'
fname2 = 'MP03-Variable_Description.xlsx'
df_variables = pd.read_excel(os.path.join(PATH, fname2))
df_data = pd.read_csv(os.path.join(PATH, fname1), encoding='ISO-8859-1') 
vars_to_keep = df_variables['Variable'].tolist()
additional_vars = ['county', 'state',
'deathspc']
vars_to_keep.extend(additional_vars)
vars_to_keep = list(set(vars_to_keep))
data_subset = df_data[vars_to_keep]
 
#Q2:
summary_statistics = data_subset.describe(include='all')
summary_statistics

#Q3:
data_subset_clean = data_subset.dropna()
data_subset_clean.head

 #Q4: 
unique_states_count = df_data['state'].nunique()
state_dummies = pd.get_dummies(data_subset_clean['state'],
drop_first=True)
data_with_dummies = pd.concat([data_subset_clean, state_dummies],
axis=1)
data_with_dummies.head()

 #Q5:
X = data_with_dummies.drop(['deathspc', 'county',
'state'], axis=1)  
y = data_with_dummies['deathspc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
(X_train.shape, X_test.shape),
(y_train.shape, y_test.shape)

 #Q6:
model = LinearRegression()
model.fit(X_train,
y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
mse_train, mse_test

#Q7:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
a = np.linspace(-2, 2, 100)  
lambdas = 10 ** a  
kf = KFold(n_splits=10, shuffle=True, random_state=25)
ridge_mse = []
lasso_mse = []
for alpha in lambdas:
    ridge_model = Ridge(alpha=alpha)
    mse_scores = cross_val_score(ridge_model, X_train_scaled, y_train, cv=kf,
 scoring='neg_mean_squared_error')
    ridge_mse.append(-np.mean(mse_scores))
for alpha in lambdas:
    lasso_model = Lasso(alpha=alpha, max_iter=10000)
    mse_scores = cross_val_score(lasso_model, X_train_scaled, y_train, cv=kf,
 scoring='neg_mean_squared_error')
    lasso_mse.append(-np.mean(mse_scores)) 

#Q7c:
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(lambdas, ridge_mse,
label='Ridge MSE',
marker='o')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.title('Ridge Regression CV MSE')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(lambdas, lasso_mse,
label='Lasso MSE', marker='o',
color='r')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.title('Lasso Regression CV MSE')
plt.legend()
plt.tight_layout()
plt.show()
 
#Q7d:
ridge_mse = np.array(ridge_mse)
lasso_mse = np.array(lasso_mse)
optimal_idx_ridge = np.argmin(ridge_mse)
optimal_idx_lasso = np.argmin(lasso_mse)
optimal_lambda_ridge = lambdas[optimal_idx_ridge]
optimal_lambda_lasso = lambdas[optimal_idx_lasso]
optimal_lambda_ridge,
optimal_lambda_lasso
 
#Q7e:
ridge_model_optimal = Ridge(alpha=optimal_lambda_ridge)
ridge_model_optimal.fit(X_train_scaled,
y_train)
lasso_model_optimal = Lasso(alpha=optimal_lambda_lasso,
max_iter=10000)
lasso_model_optimal.fit(X_train_scaled, y_train)
 
#Q8:
y_train_pred_ridge = ridge_model_optimal.predict(X_train_scaled)
y_test_pred_ridge = ridge_model_optimal.predict(X_test_scaled)
y_train_pred_lasso = lasso_model_optimal.predict(X_train_scaled)
y_test_pred_lasso = lasso_model_optimal.predict(X_test_scaled)
mse_train_ridge = mean_squared_error(y_train, y_train_pred_ridge)
mse_test_ridge = mean_squared_error(y_test, y_test_pred_ridge)
mse_train_lasso = mean_squared_error(y_train, y_train_pred_lasso)
mse_test_lasso = mean_squared_error(y_test, y_test_pred_lasso)
print("Ridge Regression:")
print(f"Training MSE: {mse_train_ridge}")
print(f"Test MSE: {mse_test_ridge}")
print("\nLassoRegression:")
print(f"Training MSE: {mse_train_lasso}")
print(f"Test MSE:{mse_test_lasso}")