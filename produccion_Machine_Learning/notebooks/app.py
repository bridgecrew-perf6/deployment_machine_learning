from statistics import stdev
import traceback
import numpy as np
import pandas as pd
import category_encoders as ce
import statistics as stats

import warnings
import sys
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")
import math
import pickle
import joblib
from joblib import load

from flask import Flask, request, jsonify
app = Flask(__name__)

columnas_seleccion = load('sel_columnas.joblib')
rf = load('apimodel.joblib')
@app.route('/predict',methods=['GET', 'POST'])
def predict():
    json_ = request.get_json(force = True)
    
    df = pd.DataFrame.from_dict(json_)

    # mean encoding

    # cargamos los diccionarios de las medias proveniente del EDA
    
    C_CONF_encoded = load('C_CONF_encoded.joblib')
    C_RCFG_encoded = load('C_RCFG_encoded.joblib')
    C_VEHS_encoded = load('C_VEHS_encoded.joblib')
    P_SEX_encoded = load('C_SEX_encoded.joblib')
    C_WTHR_encoded = load('C_WTHR_encoded.joblib')
    C_RALN_encoded = load('C_RALN_encoded.joblib')
    C_TRAF_encoded = load('C_TRAF_encoded.joblib')
    V_TYPE_encoded = load('V_TYPE_encoded.joblib')
    P_AGE_encoded = load('P_AGE_encoded.joblib')
    P_PSN_encoded = load('P_PSN_encoded.joblib')
    P_SAFE_encoded = load('P_SAFE_encoded.joblib')
    C_V_YEARS_encoded = load('C_V_YEARS_encoded.joblib')

    # Traducimos los valores introducidos en el json con los joblib anteriores

    df['C_CONF'] = df['C_CONF'].map(C_CONF_encoded)
    df['C_RCFG'] = df['C_RCFG'].map(C_RCFG_encoded)
    df['C_VEHS']= df['C_VEHS'].map(C_VEHS_encoded)
    df['P_SEX'] = df['P_SEX'].map(P_SEX_encoded)
    df['C_WTHR'] = df['C_WTHR'].map(C_WTHR_encoded)
    df['C_RALN'] = df['C_RALN'].map(C_RALN_encoded)
    df['C_TRAF'] = df['C_TRAF'].map(C_TRAF_encoded)
    df['V_TYPE'] = df['V_TYPE'].map(V_TYPE_encoded)
    df['P_AGE'] = df['P_AGE'].map(P_AGE_encoded)
    df['P_PSN'] = df['P_PSN'].map(P_PSN_encoded)
    df['P_SAFE'] = df['P_SAFE'].map(P_SAFE_encoded)
    
    # Creamos la variable C_V_YEARS

    df['C_V_YEARS']= df['C_YEAR']- df['V_YEAR']
    df['C_V_YEARS'] = df['C_V_YEARS'].astype('float')
    df = df.drop(['V_YEAR', 'C_YEAR'], axis=1)
    
    df['C_V_YEARS'] = df['C_V_YEARS'].map(C_V_YEARS_encoded)
    
    # ciclycal encoding

    columns = ['C_HOUR', 'C_MNTH', 'C_WDAY']
    
    def codificacion_ciclica(dataset, columns):
        for columna in columns:
            dataset[columna+"_norm"] = 2*math.pi*dataset[columna]/dataset[columna].max()
            dataset["cos_"+columna] = np.cos(dataset[columna+"_norm"])
            dataset["sin_"+columna] = np.sin(dataset[columna+"_norm"])
            dataset = dataset.drop([columna+"_norm"], axis=1)
        return dataset

    df['C_HOUR']= df['C_HOUR'].astype('float')
    df['C_MNTH']= df['C_MNTH'].astype('float')
    df['C_WDAY']= df['C_WDAY'].astype('float')

    df = codificacion_ciclica(df, columns)
    
    # Una vez transformadas las columnas con el ciclycal, podemos eliminar las originales
    for i in columns:
        df = (df.drop(i, axis=1))

    # Escalamos

    columnas_final=['C_VEHS', 'C_CONF', 'C_RCFG',
            'C_WTHR', 'C_RALN', 'C_TRAF', 'V_TYPE',
            'P_SEX', 'P_AGE', 'P_PSN', 'P_SAFE', 'C_V_YEARS', 'cos_C_HOUR',
            'sin_C_HOUR', 'cos_C_MNTH', 'sin_C_MNTH', 'cos_C_WDAY', 'sin_C_WDAY']
    
    
    df = df.astype('float')
    print(df.info())

    scaler = load('scaled.joblib')
    collision_scaled = pd.DataFrame(scaler.transform(df), columns=columnas_final)
    print(collision_scaled)
   
   # Calculamos las predicciones 
    prediction = rf.predict_proba(collision_scaled)
    y_pred_best = prediction[:, 1]
   
    return jsonify({'probabilidad_fallecimiento': str(y_pred_best)})

if __name__ == "__main__":
    try:
        port = int(sys.argv[1])  # This is for a command-line argument
    except:
        port = 12345  

    modelo = joblib.load("apimodel.joblib")  # Cargamos el modelo con hiperparametros
    print('Model loaded')

    model_columns = joblib.load("sel_columnas.joblib")  # Cargamos las columnas obtenidas en el Lasso
    print('Model columns loaded')

    app.run(port=port, debug=True)