from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open("flodd_prediction_model.pkl","rb"))
scaler=pickle.load(open("scaler.pkl","rb"))
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods=["POST"])
def predict():
    temp=float(request.form["temp"])
    humidity=float(request.form["humidity"])
    data=np.array([[temp,humidity]])
    data=scaler.transform(data)
    prediction=model.predict(data)
    if prediction[0]==1:
        result="Flood Likely"
    else:
        result="No Flood"
    return render_template("index.html",prediction=result)
if __name__ == "__main__":
    app.run(debug=True)