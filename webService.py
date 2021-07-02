from flask import Flask, jsonify, request


app = Flask(__name__)

@app.route('/calculoHidrogeno', methods=['POST'])
def codigoHidrogeno():
    #print(request.json)
    parameter = {
        "vacum_calc" : request.json['vacum_calc'],
        "offgas_h2" : request.json['offgas_h2'],
        "offgas_co2" : request.json['offgas_co2'],
        "hidrys_kf" : request.json['hidrys_kf'],
        "kf_temp" : request.json['kf_temp'],
        "peso_acero_olla" : request.json['peso_acero_olla'],
        "duracion_total" : request.json['duracion_total'],
        "val_vacio_presion" : request.json['val_vacio_presion']
        }
  
    from main import codigo
    cd = codigo()
    return jsonify({"message":'Datos guardados exitosamente', "resultados":cd.algoritmo(parameter)})

    
app.run(debug=True, port= 4000)
