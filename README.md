# Linear-Regression
from google.colab import files
uploaded = files.upload()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('Housing.csv') 
df.head()

X = df['area'] 
y = df['price']   

X = df[['area', 'bedrooms', 'bathrooms']]  
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))

plt.scatter(X_test['area'], y_test, color='blue') # Changed X_test to X_test['area']
plt.plot(X_test['area'], y_pred, color='red') # Changed X_test to X_test['area']
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Simple Linear Regression')
plt.show()



