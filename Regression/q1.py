import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

df=pd.read_csv('abalone.csv')

X = df.drop('Rings', axis=1) 
y = df['Rings'] 

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)

train_data = pd.concat([X_train, Y_train], axis=1)
train_data.to_csv('abalone_train.csv', index=False)

test_data = pd.concat([X_test, Y_test], axis=1)
test_data.to_csv('abalone_test.csv', index=False)



correlations=df.corr()['Rings']
highest_correlation_attr = correlations.drop('Rings').idxmax()

train_data=pd.read_csv('abalone_train.csv')
x_train=train_data[highest_correlation_attr]
y_train=train_data['Rings']

mean_x=x_train.mean()
mean_y=y_train.mean()

numerator = sum((x_train - mean_x) * (y_train - mean_y))
denominator = sum((x_train - mean_x) ** 2)
slope = numerator / denominator

intercept = mean_y - slope * mean_x

predicted_values = slope * x_train + intercept

plt.scatter(x_train,y_train,label='Training data')

plt.plot(x_train,predicted_values,color='red', label='Best-fit Line')
plt.xlabel(highest_correlation_attr)
plt.ylabel('Rings')
plt.title('Best-fit Line on Training Data')

plt.legend()
plt.show()


rmse1 = np.sqrt(mean_squared_error(train_data['Rings'], predicted_values))
print(f'the prediction accuracy on the training data using root mean squared error is {rmse1}')
x_test = test_data[highest_correlation_attr]
y_test = test_data['Rings']

predicted_values_test = slope * x_test + intercept

rmse_test = np.sqrt(mean_squared_error(y_test, predicted_values_test))
print(f'the prediction accuracy on the test data using root mean squared error is {rmse_test}')
plt.scatter(test_data['Rings'],predicted_values_test,label='Test Data')
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.show()


def polynomial_features(x, degree):
    x_poly = []
    for i in range(1, degree + 1):
        x_poly.append(x ** i)
    return pd.DataFrame(x_poly).T

degrees = [2,3,4,5]
rmse_values = []
rmse_test_value=[]

for degree in degrees:

    x_poly_train = polynomial_features(x_train, degree)
    
    XTX = x_poly_train.T.dot(x_poly_train)
    XTX_inv = pd.DataFrame(np.linalg.pinv(XTX.values), XTX.columns, XTX.index)
    XTy = x_poly_train.T.dot(y_train)
    theta = XTX_inv.dot(XTy)
    
    y_train_predicted = x_poly_train.dot(theta)
    
    squared_error = (y_train - y_train_predicted) ** 2
    mean_squared_errors = squared_error.mean()
    rmse = mean_squared_errors ** 0.5
    rmse_values.append(rmse)
    
    x_poly_test = polynomial_features(x_test, degree)
    y_test_predicted = x_poly_test.dot(theta)
    
    squared_error = (y_test - y_test_predicted) ** 2
    mean_squared_errors = squared_error.mean()
    rmse1 = mean_squared_errors ** 0.5
    rmse_test_value.append(rmse1)
    

plt.bar(degrees,rmse_values,color='blue')
plt.xlabel('Degree of polynomial')
plt.ylabel('RMSE')
plt.title('RMSE bs Degree of Polynomial')
plt.show()

   
plt.bar(degrees,rmse_test_value,color='blue')
plt.xlabel('Degree of polynomial')
plt.ylabel('RMSE')
plt.title('RMSE bs Degree of Polynomial')
plt.show()

plt.scatter(x_train,y_train,label='Training Data')
plt.plot(x_train.sort_values(),y_train_predicted.sort_values(), color='red',label='Best Fit Line')
plt.xlabel(highest_correlation_attr)
plt.ylabel('Rings')
plt.title('Best Fit Curve on Training data')
plt.legend()
plt.show()