from flask import Flask, jsonify, request


app = Flask(__name__)

@app.route('/calculoHidrogeno', methods=['POST'])
def codigoHidrogeno():
    #print(request.json)
    parameter = {
        "bool": request.json['bool'],
        "Func": request.json['Func'],
        "temp": request.json['temp'],
        "psliquido": request.json['psliquido'],
        "ppmo": request.json['ppmo'],
        "CalDolomitica_1LF" : request.json['CalDolomitica_1LF'],
        "CalSiderurgica_gruesa_1LF" : request.json['CalSiderurgica_gruesa_1LF'],
        "CalSiderurgica_fina_1LF" : request.json['CalSiderurgica_fina_1LF'],
        "AlBriqueta_1LF" : request.json['AlBriqueta_1LF'],
        "AlPosta_1LF" : request.json['AlPosta_1LF'],
        "FerroSi_gradoC_1LF" : request.json['FerroSi_gradoC_1LF'],
        "SiMnStd_gradoD_1LF" : request.json['SiMnStd_gradoD_1LF']
        }
  
    from main import codigo
    cd = codigo()
    return jsonify({"message":'Datos guardados exitosamente', "resultados":cd.algoritmo(parameter)})

    
app.run(debug=True, port= 4000)
