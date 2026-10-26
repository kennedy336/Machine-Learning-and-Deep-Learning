#Creating features
import numpy as np


X = sales_df["radio"].values
y = sales_df["sales"].values

# Reshape X
X = X.reshape(-1,1)
print(X.shape, y.shape)

#Building a linear regression model
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X, y)
predictions = reg.predict(X)

print(predictions[:5])


# visualizing the regression line
import matplotlib.pyplot as plt

plt.scatter(X, y, color="red", label="Actual Data")
plt.plot(X, predictions, color="blue", linewidth=2, label="Regression Line")

plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")
plt.legend()
plt.show()



from sklearn.metrics import root_mean_squared_error

# Compute R-squared
r_squared = reg.score(X_test, y_test)

rmse = root_mean_squared_error(y_test, y_pred)

print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = sales_df.drop(["sales", "influencer"], axis=1).values  # Drops non-feature columns
y = sales_df["sales"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate the model
reg = LinearRegression()

reg.fit(X_train, y_train)

# Make predictions (Note: predict takes only the test features, X_test)
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create X and y arrays
X = sales_df.drop(["sales", "influencer"], axis=1).values  # Drops non-feature columns
y = sales_df["sales"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg = LinearRegression()

reg.fit(X_train, y_train)

# Make predictions (Note: predict takes only the test features, X_test)
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))


# CROSS VALIDATION
from sklearn.model_selection import KFold, cross_val_score

# Create a KFold object
kf = KFold(n_splits=5, shuffle=True, random_state=42)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv=kf)

print(cv_scores)

# Regulazition L2 Ridge
from sklearn.linear_model import Ridge

alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    
    # Fit the data using training features and training targets
    ridge.fit(X_train, y_train)
    score = ridge.score(X_test, y_test)
    ridge_scores.append(score)

print(ridge_scores)

#Lasso Regression L1 REgularization
from sklearn.linear_model import Lasso

alpha = 0.3
lasso = Lasso(alpha)
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()