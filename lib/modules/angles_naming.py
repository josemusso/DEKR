import pandas as pd
import math
import numpy as np

def one_angle(data,names):
    df=data[['Second','Angle']]
    df['time']=df['Second']

    df[names]=pd.DataFrame(df["Angle"].to_list(), columns=names)
    df[names[0]]=df[names[0]].astype(float)

    df.drop(columns=['Angle', 'Second'], inplace=True)

    return df

def two_angles(data,names):
    df=data[['Second','Angle']]
    df['time']=df['Second']

    df[names]=pd.DataFrame(df["Angle"].to_list(), columns=names)
    df[names[0]]=df[names[0]].astype(float)
    df[names[1]]=df[names[1]].astype(float)

    df.drop(columns=['Angle', 'Second'], inplace=True)

    return df

def three_angles(data,names):
    df=data[['Second','Angle']]
    df['time']=df['Second']

    df[names]=pd.DataFrame(df["Angle"].to_list(), columns=names)
    df[names[0]]=df[names[0]].astype(float)
    df[names[1]]=df[names[1]].astype(float)
    df[names[2]]=df[names[2]].astype(float)

    df.drop(columns=['Angle', 'Second'], inplace=True)

    return df

def four_angles(data,names):
    df=data[['Second','Angle']]
    df['time']=df['Second']

    df[names]=pd.DataFrame(df["Angle"].to_list(), columns=names)
    df[names[0]]=df[names[0]].astype(float)
    df[names[1]]=df[names[1]].astype(float)
    df[names[2]]=df[names[2]].astype(float)
    df[names[3]]=df[names[3]].astype(float)

    df.drop(columns=['Angle', 'Second'], inplace=True)

    return df

angle_names = {
                    'M_CL'  : ['Inclinacion'],
                    'M_CP'  : ['Inclinacion'],
                    'M_CHI'  : ['Brazo'],
                    'M_CHD'  : ['Brazo'],
                    'M_SB'  : ['Brazo izquierdo', 'Brazo derecho'],
                    'M_CCO' : ['Brazo izquierdo', 'Brazo derecho'],
                    'M_RC'  : ['Rotacion'],
                    'M_LC'  : ['Brazo izquierdo','Cadera izquierda','Brazo derecho','Cadera derecha'],
                    'M_CCA' : ['Inclinacion'],
                    'M_CR'  : ['Inclinacion'],

                    'E_CLI' : ['Inclinacion'],
                    'E_CLD' : ['Inclinacion'],
                    'E_CP'  : ['Inclinacion'],
                    'E_P'   : ['Brazos','Columna'],
                    'E_ICD' : ['Brazo','Cadera'],
                    'E_ICI' : ['Brazo','Cadera'],
                    'E_FC'  : ['Cadera'],
                    'E_GD'  : ['Rodilla'],
                    'E_GI'  : ['Rodilla'],
                    'E_GED' : ['Rodilla','Cadera','Inclinacion'],
                    'E_GEI' : ['Rodilla','Cadera','Inclinacion'],

                    'D_ME'  : ['Rodilla','Cadera'],
                    'D_TG'  : ['Rodilla'],
                    'D_SL'  : ['Rodillas'],
                    'D_BJ'  : ['Brazos','Piernas'],
                    'D_AA'  : ['Desplazamiento'],
                    'D_B'   : ['Hombro'],
                    'D_SC'  : ['Rodilla','Codo'],
                    'D_RC'  : ['CD-RI','CI-RD'],

                    'A_RC'  : ['Inclinacion'],
                    'A_T'   : ['Codo'],
                    'A_A'   : ['Desplazamiento brazos'],
                    'A_FC'  : ['Cadera'],
                    'A_IF'  : ['Desplazamiento'],
                    'A_BP'  : ['Brazos','Piernas'],
                    'A_FE'  : ['Brazos','Caderas'],
                    'A_PS'  : ['Cadera','Columna'],
                    'A_PS'  : ['Cadera','Columna'],
                    'A_MT'  : ['MD_PI','MI_PD'],
                    
                    'F_SF'  : ['Rodillas','Talones'],
                    'F_SL'  : ['Rodilla','Cadera'],
                    'F_AUI' : ['Rodilla','Cadera'],
                    'F_AUD' : ['Rodilla','Cadera']
                    }

upper_exercises = {
                    'M_CL'  : True,
                    'M_CP'  : True,
                    'M_CHI'  : True,
                    'M_CHD'  : True,
                    'M_SB'  : True,
                    'M_CCO' : True,
                    'M_RC'  : True,
                    'M_LC'  : False,
                    'M_CCA' : False,
                    'M_CR'  : False,

                    'E_CLI' : True,
                    'E_CLD' : True,
                    'E_CP'  : True,
                    'E_P'   : False,
                    'E_ICD' : False,
                    'E_ICI' : False,
                    'E_FC'  : False,
                    'E_GD'  : False,
                    'E_GI'  : False,
                    'E_GED' : False,
                    'E_GEI' : False,

                    'D_ME'  : False,
                    'D_TG'  : False,
                    'D_SL'  : False,
                    'D_BJ'  : False,
                    'D_AA'  : False,
                    'D_B'   : True,
                    'D_SC'  : False,
                    'D_RC'  : False,

                    'A_RC'  : True,
                    'A_T'   : True,
                    'A_A'   : True,
                    'A_FC'  : False,
                    'A_IF'  : False,
                    'A_BP'  : False,
                    'A_FE'  : False,
                    'A_PS'  : False,
                    'A_MT'  : False,
                    
                    'F_SF'  : False,
                    'F_SL'  : False,
                    'F_AUI' : False,
                    'F_AUD' : False,

                    'L_FCODO': False
                    }
                    
separate_angles = {
                    1: one_angle,
                    2: two_angles,
                    3: three_angles,
                    4: four_angles
                }


def save_json(data,tag,path):

    try:
        angles = angle_names[tag]
        n_angles = len(angles)

        df = separate_angles[n_angles](data,angles)
        print(df)

        df.to_json(path)
        
    except:
        return None

    return df

