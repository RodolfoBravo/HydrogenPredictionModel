class codigo():

    def __init__(self):
        pass

    def algoritmo(self, parameter):
        from dataInput import Input
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import sklearn
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.neighbors import KNeighborsRegressor
        import json

        listadata = []
        #Leo los datos de la BD. Esta BD es el resultado de unir varias BD mandadas en diferentes tiempos, diferentes  variables y sin calidad
        path = "C:/Users/ECON-IT/Documents/HydrogenPredictionModel/data.xlsx"
        inputdata = Input()
        df = inputdata.leerdatos(path)
        df1 = inputdata.preprocesamiento(df)
        #Leo mis caracteristicas (X) y mi salida (Y)
        feat = [2, 3, 6, 7, 9, 18, 21, 25]
        X = df1.iloc[:,:-1].drop(columns='HidrógenoPPM')
        X = X.iloc[:,feat]
        y = df1.HidrógenoPPM

        #Normalizo los datos
        scaler_x = MinMaxScaler()
        X_n = scaler_x.fit_transform(X)
        scaler_y = MinMaxScaler()
        y_n = scaler_y.fit_transform(y.values.reshape(-1,1))

        #Separo en mi conjunto de entrenamiento y validacion
        X_train, X_test, y_train, y_test = train_test_split(X_n, y_n, test_size=0.33, random_state=42) # X_train, X_test, y_train, y_test   # (517, 8), (255, 8),(517, 1),(255, 1)

        #______________Modelo
        nn= 2
        #Defino modelo, KNN regressor
        knn= KNeighborsRegressor(n_neighbors=nn)
        #Ajusto el modelo a mis datos de entrenamiento
        knn.fit(X_train,y_train)
        #Estimo con los datos de test
        y_pred= knn.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred)

        #valor verdadero desnormalizado
        ytest = scaler_y.inverse_transform(y_test)
        #caracteristicas test desnormalizadas
        xtest = scaler_x.inverse_transform(X_test)
        #output test desnormalizado
        ytest_desn = scaler_y.inverse_transform(y_test)

        #___________________________________________________________________VALIDACION DEL MODELO____________________________________

        def normalizar_muestra(fco, matriz):
            samp_norm = []
            for i in range(0, matriz.shape[1]):
                re = (fco[i] - matriz.iloc[:, i].min()) / (matriz.iloc[:, i].max() - matriz.iloc[:, i].min())
                samp_norm.append(re)
            samp_norm=np.array(samp_norm).reshape(1,-1)
            return samp_norm


        vector_entrada_validacion={
                                    'DurationDeepVacuum_1mbar' : parameter['vacum_calc'],
                                    'OffGasH': parameter['offgas_h2'],
                                    'OffGasCO2' : parameter['offgas_co2'],
                                    'kf_value' : parameter['hidrys_kf'],
                                    'Factor_Kf_Temp': parameter['kf_temp'],
                                    'Tapping': parameter['peso_acero_olla'],
                                    'VDDuration TOTAL' : parameter['duracion_total'],
                                    'VDPressureMin' : parameter['val_vacio_presion'],
                                    }
        #separo en features y output
        X_bruto = np.array(list(vector_entrada_validacion.values()) ).reshape(1,-1) #1 x 8
        # y_bruto = np.array(list(vector_entrada_validacion.values())[-1]).reshape(1,-1) #1x1
        #normalizo
        X_norm = normalizar_muestra(X_bruto[0], X)
        #y_norm = (y_bruto - y.min()) / (y.max() - y.min())

        #prediccion
        ypred_val = knn.predict(X_norm) #realizo prediccion
        ypred_val_desn = ( (y.max() - y.min()) * ypred_val) +  y.min() #Desnormalizo prediccion

        #print(' Valor estimado: '+ str(ypred_val_desn[0][0]) )
        
        d = {"r":str(ypred_val_desn)}
        dataf = json.dumps(d)
        print (dataf)
       
        #listalistas.to_excel(r'C:/Users/ECON Tech/Desktop/ProyectoFrisa/Resultados_main.xlsx', engine='xlsxwriter')
        #print (listalistas)
        #Instruccion para que se emita un sonido indicando que la ejecucion del programa ha terminado
        return (dataf)