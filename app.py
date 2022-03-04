from statistics import mode
from flask import Flask, render_template,request
import numpy as np
from joblib import load
import os

app = Flask(__name__)

def load_clf_model():
	print(os.listdir())
	filepath = 'clf_ap.pkl'
	return load(filepath)

def predict(age=0,Gulucose=0,Insulin=0,BMI=0):
    userinp = np.array([[age, Gulucose, Insulin, BMI]])
    model_dict = load_clf_model()
    x = model_dict.get('scaler').transform(userinp)
    p = model_dict.get('classifier').predict(x)
    if p[0] == 0:
        return "NOT DIABETIC"
    else:
        return "DIABETIC"


@app.route('/', methods=['GET','POST'])
def index():
    if request.method=="POST":
      form = request.form
      age = int(form.get('age'))
      glucose = float(form.get('glucose'))
      insulin = float(form.get('insulin'))
      bmi = float(form.get('bmi'))
      result = predict(age,glucose,insulin,bmi)
      return render_template('index.html',age=age, glucose=glucose, insulin=insulin, bmi=bmi, result=result)
    return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)