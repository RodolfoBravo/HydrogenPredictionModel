import pandas as pd
class Input:
    def __init__(self):
        pass

    def leerdatos(path):
        df = pd.read_excel(path)

        #Se realiza un data preprocessing para asegurar calidad en los datos
        df.replace({'(null)': np.nan}, inplace=True)
        # Duración y VDDeepVacuunDuration son iguales
        df.drop(columns=['Temperatura_al_final_de_vacío_profundo_C', 'HidrógenoDELTA', 'ModeloHidrógeno', 'FechaProducción',
                        'VDDeepVacuunDuration'], inplace=True)
        df.drop(columns = ['Consumo_promedio_de_Ar_durante_VD_Nm3','Presión_Promedio_de_Vacío_Profundo_mbar' ], inplace=True)
        (df.HidrógenoPPM > 1.5).sum()
        df.dropna(inplace=True)
        return df

    def preprocesamiento(df):
        df1 = df.iloc[:, 6:]
        dp = (df1[df1.Prevaciado<0]).index.to_list()
        df1.drop(index = dp, inplace=True)
        df1['clase'] = np.where( df1.HidrógenoPPM < 1.5, 1,0) # 1 ok , 0 nok
        clase_gpb= df1.groupby('clase').mean()
            return df1