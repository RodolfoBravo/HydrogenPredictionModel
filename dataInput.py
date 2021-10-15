import pandas as pd
import numpy as np
class Input:
    def __init__(self):
        pass

    def leerdatos(self,path):
        df = pd.read_excel(path)
        df = df.drop(index = df[df.HidrógenoPPM == 'veo'].index.to_list())
        df.HidrógenoPPM = df.HidrógenoPPM.astype('float64')
        (df.HidrógenoPPM > 1.5).sum()
        df.dropna(inplace=True)
        df = df.loc[:,[ 'DurationDeepVacuum_1mbar', 'OffGasH', 'OffGasCO2','kf_value', 'Tapping', 'VDPressureMin','HidrógenoPPM']]
        return df

    def preprocesamiento(self,df):
        df1 = df.iloc[:, 6:]
        dp = (df1[df1.Prevaciado<0]).index.to_list()
        df1.drop(index = dp, inplace=True)
        df1['clase'] = np.where( df1.HidrógenoPPM < 1.5, 1,0) # 1 ok , 0 nok
        clase_gpb= df1.groupby('clase').mean()
        return df1