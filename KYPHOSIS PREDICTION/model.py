import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
kyphosis=pd.read_csv("kyphosis.csv")
label_encoder = preprocessing.LabelEncoder()
data['kyphosis']= label_encoder.fit_transform(data1['kyphosis'])
data['kyphosis'].unique()
x=data.drop(['kyphosis'],axis=1)
y=data(['kyphosis'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=32)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
m=regressor.fit(x_train,y_train)
pickle.dump(regressor,open('model.pkl','wb'))