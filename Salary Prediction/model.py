# Income orediction model
import pandas as pd
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
salary_data=pd.read_csv("Salary_Data.csv")
x=salary_data.drop(['Salary'],axis=1)
y=salary_data.drop(['YearsExperience'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=32)
# Since the data set is too small we are taking the whole data to train model

regressor=LinearRegression()
# Fitting the model
m=regressor.fit(x_train,y_train)
# Saving the model to disk
pickle.dump(regressor,open('model.pkl','wb'))