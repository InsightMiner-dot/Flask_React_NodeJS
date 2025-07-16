from flask import Flask, request, url_for, redirect, render_template
import pandas as pd
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


model = pickle.load(open(r"D:\2025\flask\flask_react\Building_the_backend\diabetes_model_knn.pkl", "rb"))

@app.route('/')
def use_template():
    return render_template("index.html")

@app.route('/predict', methods = ['POST','GET'])
def predict():
    input_one = request.fomr['1']
    input_two = request.fomr['2']
    input_three = request.fomr['3']
    input_four = request.fomr['4']
    input_five = request.fomr['5']
    input_six = request.fomr['6']
    input_seven = request.fomr['7']
    input_eight = request.fomr['8']
    
    
    setup_df = pd.DataFrame([pd.Series([input_one,input_two,input_three,input_four,input_five,input_six,input_seven,input_eight])])
    diabetes_prediction = model.predict_proba(setup_df)
    output = '{0:.{1}f}'.format(diabetes_prediction[0][1],2)
    output = str(float(output)*100)+'%'
    if output> str(0.5):
        return render_template('result.html',pred=f"You have the following chance of having diabetes based on our KNN model.\nProbability of having Diabetes is {output}")
    else:
        return render_template('result.html', pred = f'You have a low chance of diabetes which is currently considered safe.\nProbability of having diabetes is {output}')
    
if __name__ == '__main__':
    app.run(debug = True)
