import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, RobustScaler
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from scipy import stats

"""This code solves task of flight delay estimation. 

Used predictors:
- Departure airport
- Destination airport
- Departure Month
- Departure Day of the Week
- Flight Duration

Used models:
- Linear regression
- Linear regression with Lasso regularization
- Polynomial regression
"""

# Importing dataset
dataset = pd.read_csv('./flight_delay.csv')

                                                        # Analyzing dataset
print('\t\t\tAnalyzing dataset \n')
print(f'{dataset.head().to_string()}\n')
print(f'{dataset.shape}\n')
types = dataset.dtypes
print("Number categorical featues:", sum(types=='object'))
print(f'{types}\n')
print(dataset.describe().to_string())
print(f"{dataset['Delay'].value_counts()}\n")
print(25 * '-' + 25 * ' ' + 25 * '-')


                                                        # Label encoding
print('\t\t\t Label encoding\n')
encoder = LabelEncoder()
cat_feat_1 = ['Destination Airport']
cat_feat_2 = ['Depature Airport']
print(f'Number of unique Destination Airports {dataset[cat_feat_1].nunique()} \n')
print(f'Number of unique Destination Airports {dataset[cat_feat_2].nunique()} \n')
dataset["Destination Airport"] = encoder.fit_transform(dataset[cat_feat_1])
dataset["Depature Airport"] = encoder.fit_transform(dataset[cat_feat_2])
print(25 * '-' + 25 * ' ' + 25 * '-')


                                                        # Exctracting new features
# Departure time processing
dataset['Scheduled depature time'] = pd.to_datetime(dataset['Scheduled depature time'])
dataset['Departure Month'] = dataset['Scheduled depature time'].dt.month
dataset['Departure Day of the Week'] = dataset['Scheduled depature time'].dt.dayofweek
dataset['Departure Year'] = dataset['Scheduled depature time'].dt.year
# Arrival time processing
dataset['Scheduled arrival time'] = pd.to_datetime(dataset['Scheduled arrival time'])
# Time difference
dataset['Pre_Flight Duration'] = dataset['Scheduled arrival time'] - dataset['Scheduled depature time']
dataset['Flight Duration'] = dataset['Pre_Flight Duration'].dt.seconds / 60 
# Dropping unnessecary features
dataset.drop(['Scheduled depature time', 'Scheduled arrival time', 'Pre_Flight Duration'], axis=1, inplace = True)


                                                        # Imputing and Scaling
print('\t\t\t Imputing and Scaling\n')
def count_nans(df):
    return np.sum(np.sum(np.isnan(df)))
print("Empty cells in x_train=", count_nans(dataset))
print('No empty cells. So no Imputing is needed')

scaler = RobustScaler()
dataset["Flight Duration"] = scaler.fit_transform(dataset["Flight Duration"].values[:, None])
dataset["Depature Airport"] = scaler.fit_transform(dataset["Depature Airport"].values[:, None])
dataset["Destination Airport"] = scaler.fit_transform(dataset["Destination Airport"].values[:, None])
print(25 * '-' + 25 * ' ' + 25 * '-')


                                                        # Splitting dataset
train_data = dataset[dataset['Departure Year'] < 2018]
test_data = dataset[dataset['Departure Year'] >= 2018]
# Drop unnessecary features
train_data.drop(['Departure Year'], axis=1, inplace = True)
test_data.drop(['Departure Year'], axis=1, inplace = True)
x_test = test_data.drop('Delay', axis=1)
y_test = test_data['Delay']


                                                        # Printing
fig, ax = plt.subplots(figsize = (16,8))
ax.scatter(train_data['Flight Duration'], train_data['Delay'])
ax.set_xlabel('Flight Duration')
ax.set_ylabel('Delay')
plt.show()


                                                        # Outlier Detection and Removal
#cheching for 1 month (approximately 2% = 15000 samples)
threshold = 3
sample = train_data[0:15000]
z = np.abs(stats.zscore(sample))
out_indexes = np.where(z > threshold)
unique_indexes = np.unique(out_indexes[0])
print(f' Number of outliers: {len(unique_indexes)}')


# applying for whole dataset
z = np.abs(stats.zscore(train_data))
out_indexes = np.where(z > threshold)
unique_indexes = np.unique(out_indexes[0])
print(f' Number of outliers: {len(unique_indexes)}\n')

# Removing the outliers
train_data = train_data[(z < 3).all(axis=1)]


# Scatter plot
fig, ax = plt.subplots(figsize = (16,8))
ax.scatter(train_data['Flight Duration'], train_data['Delay'])
ax.set_xlabel('Flight Duration')
ax.set_ylabel('Delay')
plt.show()

# Deriving x_train and y_train
x_train = train_data.drop('Delay', axis=1)
y_train = y_train = train_data['Delay']

print('X_train dataframs:\n')
print(x_train.head().to_string())
print(25 * '-' + 25 * ' ' + 25 * '-')

                                                        # Linear regression
print('Linear regression\n')
lin_regr = LinearRegression()
lin_regr.fit(x_train, y_train)
print(f"Model intercept : {lin_regr.intercept_}")
print(f"Model coefficients : {lin_regr.coef_}\n")
y_pred = lin_regr.predict(x_test)

print('Root MSE for linear regression:', metrics.mean_squared_error(y_test, y_pred, squared=False))
print('MAE for linear regression:', metrics.mean_absolute_error(y_test, y_pred))
print('R2 score for linear regression', metrics.r2_score(y_test, y_pred),'\n')

train_pred = lin_regr.predict(x_train)
print('Root MSE for linear regression on test data:', metrics.mean_squared_error(y_train, train_pred, squared=False))
print('MAE for linear regression on test data:', metrics.mean_absolute_error(y_train, train_pred))
print('R2 score for linear regression on test data:', metrics.r2_score(y_train, train_pred))

print(25 * '-' + 25 * ' ' + 25 * '-')

                                                        # Lasso regression
print('Lasso regression\n')                                                       
lasso = Lasso()
gridParams = {"alpha": np.linspace(0.1, 4, 5)}
grid = GridSearchCV(lasso, gridParams, scoring='neg_mean_squared_error',
                    verbose=1, n_jobs=-1, cv=5)
grid.fit(x_train, y_train)           
print("Best params:", grid.best_params_)
print("Root MSE for lasso regression:", metrics.mean_squared_error(y_test, grid.predict(x_test), squared=False))
print("MAE for lasso regression:", metrics.mean_absolute_error(y_test, grid.predict(x_test)))
print("R2 score for lasso regression:", metrics.r2_score(y_test, grid.predict(x_test)),'\n')

pred_train = grid.predict(x_train)
print(f"Root MSE for lasso regression on train dataset:", metrics.mean_squared_error(y_train, pred_train, squared=False))
print(f"MAE for lasso regression on train dataset:", metrics.mean_absolute_error(y_train, pred_train))
print("R2 score for lasso regression on train dataset:", metrics.r2_score(y_train, pred_train),'\n')

print(25 * '-' + 25 * ' ' + 25 * '-')

                                                        # Polynomial regression
degrees = [1, 2, 3, 4, 5]
print('Polynomial regression\n')  
for degree in degrees:
    poly = PolynomialFeatures(degree)
    poly.fit(x_train)
    polyXtrain = poly.transform(x_train)
    polyXtest = poly.transform(x_test)
    lr = LinearRegression()
    lr.fit(polyXtrain, y_train)
    pred = lr.predict(polyXtest)
    print(f"Root MSE for polynomial regression with degree {degree} :", metrics.mean_squared_error(y_test, pred, squared=False))
    print(f"MAE for polynomial regression with degree {degree} :", metrics.mean_absolute_error(y_test, pred))
    print(f"R2 score  for polynomial regression with degree {degree} :", metrics.r2_score(y_test, pred),'\n')

    pred_train = lr.predict(polyXtrain)
    print(f"Root MSE for polynomial regression with degree {degree} on train dataset:", metrics.mean_squared_error(y_train, pred_train,squared=False))
    print(f"MAE for polynomial regression with degree {degree} on train dataset:", metrics.mean_absolute_error(y_train, pred_train))
    print(f"R2 score  for polynomial regression with degree {degree} :", metrics.r2_score(y_train, pred_train))
    print(25 * '-' + 25 * ' ' + 25 * '-')















