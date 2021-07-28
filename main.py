class codigo():

    def __init__(self):
        pass

    def algoritmo(self, parameter):
        from dataInput import Input
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsRegressor
        import json
    

        def desnormalizar_y(MatrizHistorica_y, y_normalizada): #tomar en cuenta que es solo una columna
            col_max = MatrizHistorica_y.max()
            col_min = MatrizHistorica_y.min()
            y_desn_list = []
            for i in range(0, len(y_normalizada)):
                y_pred_desnorm = y_normalizada[i] * (col_max - col_min) + col_min
                y_desn_list.append(y_pred_desnorm)
            y_desnormalizada_estimada= pd.DataFrame( data = np.array(y_desn_list), columns=['H_estimado'])
            return y_desnormalizada_estimada

        #Leo los datos de la BD. Esta BD es el resultado de unir varias BD mandadas en diferentes tiempos, diferentes  variables y sin calidad
        path = "C:/Users/ECON-IT/Documents/HydrogenPredictionModel - copia/Modelo Hidrogeno Ago-Dic_15-01.xlsx"# "C:/Users/Maria Luisa/OneDrive - ECON Tech/Frisa_Calculo_Hidrogeno/Modelo Hidrogeno Ago-Dic_15-01.xlsx"
        inputdata = Input()
        df = inputdata.leerdatos(path)
        df1 = inputdata.preprocesamiento(df)

        #Leo mis caracteristicas (X) y mi salida (Y)
        ''' ahora es : 'DurationDeepVacuum_1mbar', 'OffGasH', 'OffGasCO2', 'kf_value', 'Tapping', 'VDPressureMin', 'HidrógenoPPM ....Saque: #saco kf_temp, duracion total,'''
        feat = [2, 3, 6, 7, 18, 25]
        X = df1.iloc[:,:-1].drop(columns='HidrógenoPPM')
        X = X.iloc[:,feat]
        y = df1.HidrógenoPPM
        xy = pd.concat([X,y], axis=1)

        #limpiar otra vez la matriz de busqueda
        #defino los rangos de cada variable
        dict_rng = {'DurationDeepVacuum_1mbar': [xy.loc[:,'DurationDeepVacuum_1mbar' ].min(),xy.loc[:,'DurationDeepVacuum_1mbar' ].max()],
                    'OffGasH': [xy.loc[:,'OffGasH'].min(),xy.loc[:,'OffGasH'].max()],
                    'OffGasCO2':[xy.loc[:,'OffGasCO2'].min(),xy.loc[:,'OffGasCO2'].max()],
                    'kf_value': [xy.loc[:,'kf_value' ].min(),xy.loc[:, 'kf_value'].max()],
                    'Tapping':[45000,63000],
                    'VDPressureMin':[xy.loc[:,'VDPressureMin' ].min(),xy.loc[:,'VDPressureMin'].max()],
                    'HidrógenoPPM': [0.00000 ,2 ] }

        def preprocess2(xy, dict_rng):
        #Filtro los datos
            xy = xy[(xy['DurationDeepVacuum_1mbar'] >= dict_rng.get('DurationDeepVacuum_1mbar')[0]) & (xy['DurationDeepVacuum_1mbar']<= dict_rng.get('DurationDeepVacuum_1mbar')[1] )]
            xy = xy[(xy['OffGasH'] >= dict_rng.get('OffGasH')[0] ) & (xy['OffGasH'] <= dict_rng.get('OffGasH')[1] )] #754
            xy = xy[(xy['OffGasCO2']>= dict_rng.get('OffGasCO2')[0] ) & (xy['OffGasCO2']<= dict_rng.get('OffGasCO2')[1] )]
            xy = xy[(xy['kf_value'] >= dict_rng.get('kf_value')[0]) & (xy['kf_value']<= dict_rng.get('kf_value')[1])]
            xy = xy[(xy['Tapping'] >= dict_rng.get('Tapping' )[0])  & (xy['Tapping'] <= dict_rng.get('Tapping' )[1])]
            xy = xy[(xy['VDPressureMin']>= dict_rng.get('VDPressureMin' )[0]) & (xy['VDPressureMin']<= dict_rng.get('VDPressureMin')[1])]
            xy = xy[(xy['HidrógenoPPM'] > dict_rng.get('HidrógenoPPM')[0]) & (xy['HidrógenoPPM']<= dict_rng.get('HidrógenoPPM')[1])]#733
            return xy
        xy = preprocess2(xy, dict_rng)

        #____________________________________ MODELO
        #Defino mi X y y
        X = xy.iloc[:, :-1]
        y = xy.iloc[:, -1]

        #Normalizo los datos
        def normalizar_df(dataframe1):
            norm_df = []
            i,j = 0, 0
            for j in range(0, dataframe1.shape[1]):
                columna = dataframe1.iloc[:,j]
                max_col = columna.max()
                min_col = columna.min()
                for i in range(0, dataframe1.shape[0]):
                    norm_muestra = (dataframe1.iloc[i,j] - min_col) / ( max_col - min_col)
                    norm_df.append(norm_muestra)
            df_normalizado = pd.DataFrame(data = np.array(norm_df).reshape(dataframe1.shape[1], dataframe1.shape[0])).transpose() #, columns= dataframe1.columns.to_list())
            return df_normalizado

        X_n = normalizar_df(X)
        y_n = (y - y.min() ) /( y.max() - y.min())
   
        #Separo en mi conjunto de entrenamiento y validacion
        X_train, X_test, y_train, y_test = train_test_split(X_n, y_n, test_size=0.2, random_state=42) # X_train, X_test, y_train, y_test   # (517, 8), (255, 8),(517, 1),(255, 1)

        #______________Modelo
        nn= 2
        #Defino modelo, KNN regressor
        knn= KNeighborsRegressor(n_neighbors=nn)
        knn.fit(X_train,y_train) #Ajusto el modelo a mis datos de entrenamiento
        y_pred= knn.predict(X_test) #Estimo con los datos de test
        y_pred = desnormalizar_y(y, y_pred)
        ytest = y_test * ( y.max() - y.min() ) + y.min()  #valor verdadero desnormalizado

        #_______________________Leer datos de un evento
        vector_entrada_validacion={
                                    'Vacuum_2mbar_min' : parameter['vacum_calc'],
                                    'offgas_h2':parameter['offgas_h2'],
                                    'offgas_co2': parameter['offgas_co2'],
                                    'hidrys_Kf' :parameter['hidrys_kf'],
                                    'peso_acero_olla':parameter['peso_acero_olla'],
                                    'val_vacio_presion_waste_line' :parameter['val_vacio_presion'],
                                    }
        x_nueva = pd.DataFrame.from_dict([vector_entrada_validacion] )

        def filtro_data(data_new_):
            pasaron_filtro = []
            no_pasaron_filtro = []
            muestra = data_new_
            if (muestra.hidrys_Kf.values[0] !=0):
                if (
                    #condicion de vacio profundo
                    (muestra.Vacuum_2mbar_min.values >= dict_rng.get('DurationDeepVacuum_1mbar')[0]) and
                    (muestra.Vacuum_2mbar_min.values <= dict_rng.get('DurationDeepVacuum_1mbar')[1]) and
                    #condicion de offGasH
                    (muestra.offgas_h2.values >= dict_rng.get('OffGasH')[0]) and
                    (muestra.offgas_h2.values <= dict_rng.get('OffGasH')[1]) and
                    #condicion offGasCo2
                    (muestra.offgas_co2.values >= dict_rng.get('OffGasCO2')[0]) and
                    (muestra.offgas_co2.values <= dict_rng.get('OffGasCO2')[1]) and
                    #condicion kf_value
                    (muestra.hidrys_Kf.values >= dict_rng.get('kf_value')[0]) and
                    (muestra.hidrys_Kf.values <= dict_rng.get('kf_value')[1]) and
                    #condicion 'Tapping'
                    (muestra.peso_acero_olla.values >= dict_rng.get('Tapping')[0]) and
                    (muestra.peso_acero_olla.values <= dict_rng.get('Tapping')[1]) and
                    #condicion 'VDPressureMin'
                    (muestra.val_vacio_presion_waste_line.values >= dict_rng.get('VDPressureMin')[0]) and
                    (muestra.val_vacio_presion_waste_line.values <= dict_rng.get('VDPressureMin')[1])
                    ):
                    pasaron_filtro.append(muestra)
                else:
                    no_pasaron_filtro.append(muestra)
            else:
                no_pasaron_filtro.append(muestra)
            return pasaron_filtro, no_pasaron_filtro
        pasaron_filtro, no_pasaron_filtro = filtro_data(x_nueva)     

        def realizar_estimacion():
            if len(pasaron_filtro) != 0:
                #print('Realizar estimacion. El vector se encuentra dentro de los rangos de la matriz de busqueda')

                # los que pasaron filtro, se les normaliza
                def norm_muestra_filtrada(MatrizdeBusqueda,
                                        muestra):  # Las caracteristicas de la muestra debe estar en el mismo orden de la matriz de busqueda
                    x_maria = muestra.values.ravel()
                    x_moni = (x_maria - MatrizdeBusqueda.min().ravel()) / (
                                MatrizdeBusqueda.max().ravel() - MatrizdeBusqueda.min().ravel())
                    return x_moni

                muestra_normalizada_filtrada = norm_muestra_filtrada(X, pasaron_filtro[0])
                y_new_pred = knn.predict(muestra_normalizada_filtrada.reshape(1, -1))
                y_desnormalizada_estimada = desnormalizar_y(y, y_new_pred)

              #  print(y_desnormalizada_estimada )
                result = y_desnormalizada_estimada.values.ravel()
                print (result[0])
                return result[0]
            else:
                msg = 'El vector se encuentra fuera de rango de la matriz de busqueda, y por tanto no paso los filtros'
                print (msg)
                return msg
              
        y_estimada_final = realizar_estimacion()

        d = {"r": str(y_estimada_final)}
        data = json.dumps(d)
        print (data)
        return (data)