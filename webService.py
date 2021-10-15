from performance import Modelperformance
from flask import Flask, jsonify, request
import pandas as pd


app_ml = Flask(__name__)

@app_ml.route('/calculoHidrogeno', methods=['POST'])
def codigoHidrogeno():
    #print(request.json)
    parameter = {
        "vacum_calc" : request.json['vacum_calc'],
        "offgas_h2" : request.json['offgas_h2'],
        "offgas_co2" : request.json['offgas_co2'],
        "hidrys_kf" : request.json['hidrys_kf'],
        "peso_acero_olla" : request.json['peso_acero_olla'],
        "val_vacio_presion" : request.json['val_vacio_presion']
        }
  
    from main import codigo
    cd = codigo()
    return jsonify({"message":'Datos guardados exitosamente', "resultados":cd.algoritmo(parameter)})

@app_ml.route('/performance1', methods=['POST'])
def performance():
    path_osv = request.json["path"]
    print(path_osv)
    from performance import Modelperformance
    path_output_summary_vector = path_osv
    outputsummaryvector= pd.read_csv(path_output_summary_vector)
    bd_filtrada = Modelperformance.filtro(outputsummaryvector)
    bd_filtrada_json = bd_filtrada.to_json()
    resultados, wrong, yield_final , mae, model_use= Modelperformance.metricas(bd_filtrada ,outputsummaryvector)
    resultados = resultados.to_json()

    return jsonify({"message":'Datos guardados exitosamente', "resultados": resultados, "wrong":wrong,"yield_": yield_final_,"MAE":mae,"model_use":model_use})


app_ml.run(debug=True, port= 4000)