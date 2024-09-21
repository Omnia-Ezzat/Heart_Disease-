from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__,template_folder="templets")
model=pickle.load(open('notebooks/RF_model.pkl','rb'))
scaler = pickle.load(open('notebooks/scaler (1).pkl', 'rb'))

input_names=[
'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak','Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'
]
cat=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
num=['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['post'])
def predict():
  features=[]
  for col in input_names:
     value=request.form.get(col)
     
     if col in cat:
        e=pickle.load(open('notebooks/{}_encoding_mapping.pkl'.format(col),'rb'))
        v = e.transform(np.array([[value]]))[0][0]
        features.append(v)
     else:
       features.append(value)
  
    # Convert features to numpy array and scale numerical features
  features = np.array(features, dtype='object').reshape(1, -1)

    # Scale only numerical features (non-categorical)
  features[:, [input_names.index(col) for col in num]] = scaler.transform(
        features[:, [input_names.index(col) for col in num]]
    )

  y_pred=model.predict(features)
  output=' '
  if y_pred==1:
         output='May has Heart Disease '
  else:
        output='May not have Heart Disease'

  return render_template('result.html',prediction_text=output)    



if __name__=="__main__":
    app.run(debug=True)

