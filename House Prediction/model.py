import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
house_prices=pd.read_csv("homeprices (1).csv")
x=house_prices.drop(['price'],axis=1)
y=house_prices.drop(['area'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=32)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
m=regressor.fit(x_train,y_train)
pickle.dump(regressor,open('model.pkl','wb'))