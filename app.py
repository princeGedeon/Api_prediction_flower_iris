from flask import Flask,render_template,session,url_for
import numpy as np
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from tensorflow.keras.models import load_model
import joblib
app = Flask(__name__)

app.config["SECRET_KEY"]="djsdprincdenksc"

class FlowerForm(FlaskForm):
    sep_lenght=TextField("Longueur du sepale")
    sep_width=TextField("Largeur du sepale")
    pet_lenght = TextField("Longueur du petale")
    pet_width = TextField("Largeur du petale")
    submit=SubmitField("Analyser")
def return_prediction(model, scaler, flower_example):
    s_len = flower_example['sepal_length']
    s_wid = flower_example['sepal_width']
    p_len = flower_example['petal_length']
    p_wid = flower_example['petal_width']
    flower = [[s_len, s_wid, p_len, p_wid]]
    flower = scaler.transform(flower)
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    class_ind = np.argmax(model.predict(flower), axis=1)

    return classes[class_ind]

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

flower_model=load_model("final_iris_model.h5")
flower_scaler=joblib.load("iris_scaler.pkl")

@app.route('/api/flower',methods=['POST'])
def flower_prediction():
    content=request.json 
    result=return_prediction(flower_model,flower_scaler,content)
    return jsonify(result)


if __name__ == '__main__':
    app.run()
