from flask import Flask , request , jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib


def return_prediction(model,scaler,sample_json):
    
    # For larger data features, you should probably write a for loop
    # That builds out this array for you

   # get the input data from the JSON
    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']
    
    # make the input as list to fed it to the model
    flower = [[s_len,s_wid,p_len,p_wid]]
    # Scale data
    #from warnings import simplefilter
    # ignore all future warnings
    #simplefilter(action='ignore', category=FutureWarning)    
    flower = scaler.transform(flower)

    classes = np.array(['setosa', 'versicolor', 'virginica'])
    
    pred = model.predict(flower,verbose=0)
    class_ind = pred.argmax()
    res = classes[class_ind]
    return res

app = Flask(__name__)

@app.route("/")
def index():
	return '<h1>Flask APP is running</h1>'


flower_model = load_model("final_iris_model.h5")
flower_scaler = joblib.load("iris_scaler.pkl")

@app.route('/api/flower',methods=['POST'])
def flower_prediction():
	content = request.json
	results = return_prediction(flower_model,flower_scaler,content)
	return jsonify(results)


if __name__=='__main__':
	app.run()