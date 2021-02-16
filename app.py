from flask import Flask , render_template , request
import pickle
import numpy as np


app = Flask(__name__)
model=pickle.load(open("model.pkl" , 'rb'))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=["GET" , "POST"])
def predict():
    
    ram = float(request.form['ram'])
    battery_power = float(request.form['battery_power'])
    int_memory = float(request.form['int_memory'])
    mob_weight = float(request.form['mob_weight'])
    px_height = float(request.form['px_height'])
    px_width = float(request.form['px_width'])
    
    pred_agrs = [ram , battery_power , int_memory , mob_weight , px_height , px_width]
                       
    pred_agrs_arr = np.array(pred_agrs)
    pred_agrs_arr = pred_agrs_arr.reshape(1 , -1)
            
    model_pred = model.predict(pred_agrs_arr)
    model_pred = round(float(model_pred))
    
    if model_pred==0:
        return render_template('predict.html' , prediction = "Mobile price will be below 10000")
    elif model_pred==1:
        return render_template('predict.html' , prediction = "Mobile price will be between 10000-20000")
    elif model_pred==2:
        return render_template('predict.html' , prediction = "Mobile price will be between 20000-30000")
    else:
        return render_template('predict.html' , prediction = "Mobile price will be above 30000")
    


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




