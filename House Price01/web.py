from flask import Flask,render_template,request
import numpy as np
import pickle
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_templates("home.html")
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=np.array(int_features).reshape(1,2)
    print(final_features)
    return render_template("result.html",prediction_text="Congrats!!!....you are eligible for the salary of Rs.{}")

if __name__=='__main__':
    app.run(port=8000)