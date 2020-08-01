import numpy as np
from flask import Flask,request,render_template
import pickle
import pandas as pd
from sklearn.externals import joblib
app = Flask(__name__)
model=joblib.load('breast_cancer_detector.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

        intput_feature=[float(x) for x in request.form.values()]
        feature_values=[np.array(intput_feature)]
        feature_name=['mean radius','mean texture','mean perimeter','mean area','mean smoothness','mean compactness','mean concavity',
        'mean concave points','mean symmetry','mean fractal dimension',
        'radius error','texture error','perimeter error','area error',
        'smoothness error','compactness error','concavity error',
        'concave points error','symmetry error','fractal dimension error',
        'worst radius','worst texture','worst perimeter','worst area',
        'worst smoothness','worst compactness','worst concavity',
        'worst concave points','worst symmetry','worst fractal dimension']
        df=pd.DataFrame(feature_values,columns=feature_name)
        output=model.predict(df)
        second_output=bool(output)
        if second_output==0:
            res_val="**Yes, Breast Cancer **"
        else:
            res_val="** No, Breast Cancer **"
        return render_template('index.html',prediction_text='This Human have cancer (True / False ):  {}'.format(second_output))


if __name__=="__main__":
        app.run(debug=True)