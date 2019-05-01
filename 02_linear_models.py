# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import io
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import lognorm
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pickle 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

CSV_URL = 'https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2015-09.csv'
s = requests.get(CSV_URL).content
data = pd.read_csv(io.StringIO(s.decode('utf-8')))


data['lpep_pickup_datetime'] = pd.to_datetime(data['lpep_pickup_datetime'])
data['pickup_hour'] = data['lpep_pickup_datetime'].dt.hour


nrow, ncol = data.shape



    
Trips_credit = data.loc[(data['Payment_type']==1),:]
charge = ['Total_amount', 'Tip_amount', 'Tolls_amount', 
          'Fare_amount', 'Extra', 'MTA_tax', 'improvement_surcharge']

Trips_credit[charge] = Trips_credit[charge].abs()

Trips_credit_valid = Trips_credit.loc[(Trips_credit['Total_amount'] >= 2.50)]

Trips_credit_valid = Trips_credit_valid.loc[Trips_credit_valid['RateCodeID'] < 5,:]

Trips_credit_valid['Tip_precent'] = Trips_credit_valid['Tip_amount'] / (Trips_credit_valid['Total_amount'] - Trips_credit_valid['Tip_amount'])


def time_features(df, col):
    df['weekofyear'] = df[col].dt.weekofyear
    df['dayofweek'] = df[col].dt.dayofweek
    df['weekend'] = (df[col].dt.weekday >=5).astype(int)
    # df['hour'] = df[col].dt.hour

time_features(Trips_credit_valid, 'lpep_pickup_datetime')

Trips_credit_valid['Lpep_dropoff_datetime'] = pd.to_datetime(Trips_credit_valid['Lpep_dropoff_datetime'])
Trips_credit_valid['trip_time'] = (Trips_credit_valid['Lpep_dropoff_datetime'] - Trips_credit_valid['lpep_pickup_datetime']).dt.seconds/60.0
Trips_credit_valid['avg_speed'] = Trips_credit_valid['Trip_distance'] / Trips_credit_valid['trip_time']

med_speed = Trips_credit_valid['avg_speed'].quantile(0.5)
Trips_credit_valid['avg_speed'] = Trips_credit_valid['avg_speed'].fillna(med_speed)
Trips_credit_valid['avg_speed'] = Trips_credit_valid['avg_speed'].replace(np.inf, med_speed)
Trips_credit_valid['avg_speed'] = Trips_credit_valid['avg_speed'].fillna(med_speed)

Trips_credit_valid['Tolls'] = Trips_credit_valid['Tolls_amount'].apply(lambda x: 0 if x == 0 else 1)
drop_log_med = np.median(Trips_credit_valid['Dropoff_longitude'])
drop_lat_med = np.median(Trips_credit_valid['Dropoff_latitude'])
Trips_credit_valid['pick_loc'] = (Trips_credit_valid['Dropoff_longitude'] - drop_log_med) ** 2 + (Trips_credit_valid['Dropoff_latitude'] - drop_lat_med) ** 2
pick_log_med = np.median(Trips_credit_valid['Pickup_longitude'])
pick_lat_med = np.median(Trips_credit_valid['Pickup_latitude'])
Trips_credit_valid['drop_loc'] = (Trips_credit_valid['Pickup_longitude'] - pick_log_med) ** 2 + (Trips_credit_valid['Pickup_latitude'] - pick_lat_med) ** 2
Trips_credit_valid['Passenger_count'] = Trips_credit_valid['Passenger_count'].apply(lambda x: 1 if x == 0 else x)


y = Trips_credit_valid['Tip_precent']
col_to_drop = ['Store_and_fwd_flag', 'Tip_amount', 'Tip_precent', 'lpep_pickup_datetime', 'Lpep_dropoff_datetime', 'Pickup_longitude',
       'Pickup_latitude', 'Dropoff_longitude', 'Dropoff_latitude', 'Ehail_fee']
for col in col_to_drop:
    Trips_credit_valid.drop([col], axis=1, inplace=True)
    


def convert_to_categorical(data):
    for col in data.columns:
        if col in ['VendorID', 'RateCodeID', 'Extra', 'MTA_tax']:
            data[col] = data[col].astype('category')
    return data
Trips_credit = convert_to_categorical(Trips_credit_valid)

a = ['VendorID', 'RateCodeID', 'Extra', 'MTA_tax']
b = list(Trips_credit.columns)




enc = OneHotEncoder(categorical_features=[ b.index(x) if x in b else None for x in a ])
# enc.fit(Trips_credit)

X_transform = enc.fit_transform(Trips_credit)

X_train, X_test, y_train, y_test = train_test_split(X_transform, y, test_size=0.2, random_state=100)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# prepare a range of alpha values to test
alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
# create and fit a ridge regression model, testing each alpha
model = linear_model.Lasso(normalize=True)
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas), cv=4)
grid.fit(X_train, y_train)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.alpha)




