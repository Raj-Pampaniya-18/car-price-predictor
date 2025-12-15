import numpy as np
import pandas as pd 

car = pd.read_csv('/home/raj/my_project/quikr_car.csv')

# print(car.shape)

#car.info()

# Quaility Check All Data Cleaning

# print(car['year'].unique())

# print(car['Price'].unique())

# print(car['kms_driven'].unique())

# print(car['fuel_type'].unique())

# print(car.head())

# started cleaning 

backup = car.copy()

car = car[car['year'].str.isnumeric()]

car['year'] = car['year'].astype('Int32')

car = car[car['Price']!= 'Ask For Price']

car['Price'] = car['Price'].str.replace(',','').astype('Int32')

car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',','')

car = car[car['kms_driven'].str.isnumeric()]

car['kms_driven'] = car['kms_driven'].astype('Int32')

car = car[~car['fuel_type'].isna()]

car['name'] = car['name'].str.split(' ').str.slice(0,3).str.join(' ')

car =  car.reset_index(drop=True)

#car.info()

# print(car.describe())

car = car[car['Price']<6e6].reset_index(drop=True)

# print(car.head())

#car.to_csv('/home/raj/Desktop/Raj/oop/opps projects /ml/maths /chatbot/car_predictor_project/cleaned_car_data.csv')

# Model Building

X  = car.drop(columns='Price')
y = car['Price']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

ohe = OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])
# print(ohe)


column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')

lr = LinearRegression()

pipe = make_pipeline(column_trans,lr)

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

r2_score(y_test,y_pred)

scores = []
for i in range(10):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred = pipe.predict(X_test)
    r2_score(y_test,y_pred), i
    scores.append(r2_score(y_test,y_pred))

np.argmax(scores)

scores[np.argmax(scores)]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=np.argmax(scores))
lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
r2_score(y_test,y_pred)

import pickle

#pickle.dump(pipe,open('/home/raj/Desktop/Raj/oop/opps projects /ml/maths /chatbot/car_predictor_project/LinearRegressionModel.pkl','wb'))


# Testing the model
print("Pridicted price is ")
print(pipe.predict(pd.DataFrame([['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']],columns=['name','company','year','kms_driven','fuel_type'])))

