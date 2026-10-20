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