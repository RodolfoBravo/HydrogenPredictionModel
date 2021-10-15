import pandas as pd
import numpy as np
from dicc_dominio import dicc_domain
from sklearn.metrics import confusion_matrix

class Modelperformance:
    def __init__(self):
        pass
    
    

    def filtro(xy):
        dict_rng= dicc_domain.dict_rng
        #Filtro los datos
        xy = xy[(xy['Vacuum_2mbar_min'] >= dict_rng.get('DurationDeepVacuum_1mbar')[0]) & (xy['Vacuum_2mbar_min']<= dict_rng.get('DurationDeepVacuum_1mbar')[1] )]
        xy = xy[(xy['offgas_h2'] >= dict_rng.get('OffGasH')[0] ) & (xy['offgas_h2'] <= dict_rng.get('OffGasH')[1] )] #754
        xy = xy[(xy['offgas_co2']>= dict_rng.get('OffGasCO2')[0] ) & (xy['offgas_co2']<= dict_rng.get('OffGasCO2')[1] )]
        xy = xy[(xy['hidrys_Kf'] >= dict_rng.get('kf_value')[0]) & (xy['hidrys_Kf']<= dict_rng.get('kf_value')[1])]
        xy = xy[(xy['peso_acero_olla'] >= dict_rng.get('Tapping' )[0])  & (xy['peso_acero_olla'] <= dict_rng.get('Tapping' )[1])]
        xy = xy[(xy['val_vacio_presion_waste_line']>= dict_rng.get('VDPressureMin' )[0]) & (xy['val_vacio_presion_waste_line']<= dict_rng.get('VDPressureMin')[1])]
        xy = xy[(xy['vesuvius_ppm'] > dict_rng.get('HidrógenoPPM')[0]) & (xy['vesuvius_ppm']<= dict_rng.get('HidrógenoPPM')[1])]#733
        return xy
    
    def metricas(dataframe, dataframeoriginal):
        x = dataframe.loc[: ,['Vacuum_2mbar_min', 'offgas_h2', 'offgas_co2','hidrys_Kf' ,'peso_acero_olla','val_vacio_presion_waste_line']]
        y_real = dataframe.loc[: , ['hydris_ppm','vesuvius_ppm']]
        y_pred = dataframe.loc[: ,'result_model']
        #resultados
        resultados = dataframe.loc[:, ['t_stamp','numero_colada', 'vesuvius_ppm','result_model']]
        resultados = resultados.sort_values(by='vesuvius_ppm')
        resultados['residuo']= ( resultados.vesuvius_ppm - resultados.result_model)
        resultados['error_abs']= abs( resultados.vesuvius_ppm - resultados.result_model)
        resultados['Hreal_lesseq1'] = [1 if i <= 1 else 0 for i in resultados.vesuvius_ppm ]
        resultados['Hpred_lesseq1'] = [1 if i <= 1 else 0 for i in resultados.result_model]
        #metricas
        conf_mat = confusion_matrix(resultados.Hreal_lesseq1,resultados.Hpred_lesseq1 ) #[ [ 'TP', 'FN' ] , ['FP',  'TN' ]]
        mae = abs( resultados.vesuvius_ppm - resultados.result_model).mean()
        wrong =  (conf_mat[1][0] + conf_mat[0][1]) / len(resultados)
        yield_ = (1- wrong) * 100
        
        model_use= ( len(dataframe)/len(dataframeoriginal) ) *100
        return resultados, conf_mat, wrong, yield_, mae, model_use
    
    