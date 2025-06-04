
#Consider the data about house like size,
#          no of bedrooms,and location using this predict their selling price
# *can be used for project*



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data = pd.read_csv('house_prices.csv')  # Replace with actual dataset
print(data.head())
X = data[['area', 'bedrooms', 'bathrooms']]   # Input features
y = data['price']       # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:",rmse)
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
# plt.plot(y_test, y_pred)
# plt.bar(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()