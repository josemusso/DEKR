import pandas as pd
import numpy as np
import math
from scipy.signal import argrelextrema

#1. brazos
#2. piernas


def score(data):
    score=True
    ## Calification rules
    max_border_1 = 230          # Max threshold for valid point
    min_border_1 = 0            # Min threshold for valid point

    max_border_2 = 230          # Max threshold for valid point
    min_border_2 = 0            # Min threshold for valid point

    max_threshold = 80          # Threshold for valid peak
    min_ang = 80                # Threshold for great peak

    n_exp_rep = 8               # Expected repetitions
    n_min_rep = 6               # Min repetitions

    #Read data
    df = data

    # Order data
    df=df[['Second','Angle']]
    df[['ang_1','ang_2']]=pd.DataFrame(df["Angle"].to_list(), columns=['ang_1','ang_2'])
    df['time']=df['Second']
    df.drop(columns=['Angle', 'Second'], inplace=True)

    df['ang_1']=df['ang_1'].astype(float)
    df['ang_2']=df['ang_2'].astype(float)

    # Calculate aprox frame rate and set number of frames to compare
    max_window=1                                            # Time window (seconds)
    df['frame_t'] = abs(df.time - df.time.shift(1))         # Get time difference for every frame
    mean_ftime=df.frame_t.median()                          # Get average difference
    fps=round(1/mean_ftime,2)                               # Aproximate FPS
    n=max_window/mean_ftime                                 # Number of frames in the time window
    n=int(n)                                # Round number of frames

    # Fix unvalid points
    df.loc[df['ang_1'] > max_border_1, 'ang_1'] = df.ang_1.mean()
    df.loc[df['ang_1'] < min_border_1, 'ang_1'] = df.ang_1.mean()

    df.loc[df['ang_2'] > max_border_2, 'ang_2'] = df.ang_2.mean()
    df.loc[df['ang_2'] < min_border_2, 'ang_2'] = df.ang_2.mean()

    #Save original stats
    df_o_stats=df.describe().astype(float).round(3).to_dict()


    # Search for max and min points
    df['min_1'] = df.iloc[argrelextrema(df.ang_1.values, np.less_equal,
                        order=n)[0]]['ang_1']
    df['max_1'] = df.iloc[argrelextrema(df.ang_1.values, np.greater_equal,
                        order=n)[0]]['ang_1']

    # Keep only points on valid threshold.
    df.loc[df['min_1'] >= min_ang, 'min_1'] = np.nan
    df.loc[df['max_1'] <= max_threshold, 'max_1'] = np.nan


    # Search for max and min points
    df['min_2'] = df.iloc[argrelextrema(df.ang_2.values, np.less_equal,
                        order=n)[0]]['ang_2']
    df['max_2'] = df.iloc[argrelextrema(df.ang_2.values, np.greater_equal,
                        order=n)[0]]['ang_2']

    # Keep only points on valid threshold.
    df.loc[df['min_2'] >= min_ang, 'min_2'] = np.nan
    df.loc[df['max_2'] <= max_threshold, 'max_2'] = np.nan

    window = 0.5 #seconds
    ## Take one point per maxima
    df_min_1 = df[df['min_1'].notnull()].copy()
    df_max_1 = df[df['max_1'].notnull()].copy()

    # Mark points to take. Breach between points must be greater than the window.
    df_min_1['valid'] = df_min_1.time[abs(df_min_1.time - df_min_1.time.shift(-1,fill_value=0))>window]
    df_max_1['valid'] = df_max_1.time[abs(df_max_1.time - df_max_1.time.shift(-1,fill_value=0))>window]

    df_min_1 = df_min_1[df_min_1['valid'].notnull()]
    df_max_1 = df_max_1[df_max_1['valid'].notnull()]

    df_f_1= df.copy()

    df_f_1['min_1'] = np.nan
    df_f_1['max_1'] = np.nan

    # Save the interest points on original dataframe
    df_f_1['min_1'] = df.iloc[df_min_1.index]['min_1']
    df_f_1['max_1'] = df.iloc[df_max_1.index]['max_1']

    df_1 = df_f_1.copy()

    ## Take one point per maxima
    df_min_2 = df[df['min_2'].notnull()].copy()
    df_max_2 = df[df['max_2'].notnull()].copy()

    # Mark points to take. Breach between points must be greater than the window.
    df_min_2['valid'] = df_min_2.time[abs(df_min_2.time - df_min_2.time.shift(-1,fill_value=0))>window]
    df_max_2['valid'] = df_max_2.time[abs(df_max_2.time - df_max_2.time.shift(-1,fill_value=0))>window]

    df_min_2 = df_min_2[df_min_2['valid'].notnull()]
    df_max_2 = df_max_2[df_max_2['valid'].notnull()]

    df_f_2= df.copy()

    df_f_2['min_2'] = np.nan
    df_f_2['max_2'] = np.nan

    # Save the interest points on original dataframe
    df_f_2['min_2'] = df.iloc[df_min_2.index]['min_2']
    df_f_2['max_2'] = df.iloc[df_max_2.index]['max_2']

    df_2 = df_f_2.copy()

    # Separate angles DF ( to clean tails)
    df_1=df_1[['time','ang_1', 'max_1','min_1']]
    df_2=df_2[['time','ang_2', 'max_2','min_2']]

    ## Cut tails
    try:
        # Angle 1
        df_s = df_1[df_1['min_1'].notnull()].copy()
        start=df_s.index[0]
        end=df_s.index[-1]
        df_1 = df_1[start:end+1]
    except:
        pass
    try:
        # Angle 2
        df_s = df_2[df_2['min_2'].notnull()].copy()
        start=df_s.index[0]
        end=df_s.index[-1]
        df_2 = df_2[start:end+1]
    except:
        pass


    # Number of peaks.
    n_min_1 = df_1[df_1['min_1'].notnull()].min_1.count()
    n_max_1 = df_1[df_1['max_1'].notnull()].max_1.count()

    # Peaks median.
    med_min_1 = df_1.min_1.median()
    med_max_1 = df_1.max_1.median()

    # Number of peaks.
    n_min_2 = df_2[df_2['min_2'].notnull()].min_2.count()
    n_max_2 = df_2[df_2['max_2'].notnull()].max_2.count()

    # Peaks median.
    med_min_2 = df_2.min_2.median()
    med_max_2 = df_2.max_2.median()

    #Min points frequency
    df_min_1 = df_1[df_1['min_1'].notnull()]
    df_min_1['dif'] = abs(df_min_1.time - df_min_1.time.shift(-1))
    freq_min_1 = 1/df_min_1.dif.mean()

    #Max points frequency
    df_max_1 = df_1[df_1['max_1'].notnull()]
    df_max_1['dif'] = abs(df_max_1.time - df_max_1.time.shift(-1))
    freq_max_1 = 1/df_max_1.dif.mean()

    #Min points frequency
    df_min_2 = df_2[df_2['min_2'].notnull()]
    df_min_2['dif'] = abs(df_min_2.time - df_min_2.time.shift(-1))
    freq_min_2 = 1/df_min_2.dif.mean()

    #Max points frequency
    df_max_2 = df_2[df_2['max_2'].notnull()]
    df_max_2['dif'] = abs(df_max_2.time - df_max_2.time.shift(-1))
    freq_max_2 = 1/df_max_2.dif.mean()

    # Results  1.
    print('N. minimos: %d' % (n_min_1))
    print('Angulo Promedio max.: %.2f / Angulo Promedio min.: %.2f' %(med_max_1,med_min_1))
    print('Freq. maximos: %.2f / Freq. minimos: %.2f' % (freq_max_1,freq_min_1))

    # Results 2
    print('N. minimos: %d' % (n_min_2))
    print('Angulo Promedio max.: %.2f / Angulo Promedio min.: %.2f' %(med_max_2,med_min_2))
    print('Freq. maximos: %.2f / Freq. minimos: %.2f' % (freq_max_2,freq_min_2))

    #Save final stats
    df_1_stats=df_1.describe().astype(float).round(3).to_dict()
    df_2_stats=df_2.describe().astype(float).round(3).to_dict()

    # Save frquency on stats
    df_1_stats['min_1']['freq']=round(freq_min_1,3)
    df_1_stats['max_1']['freq']=round(freq_max_1,3)
    df_2_stats['min_2']['freq']=round(freq_min_2,3)
    df_2_stats['max_2']['freq']=round(freq_max_2,3)

    score=True
    """ med_max_1=150
    med_max_2=170 """

    rep_rec=[]
    angle_rec=[]
    recommendations=[]

    if(n_min_1<=n_min_2):
        n_rep=n_min_1
    else:
        n_rep=n_min_2

    if (n_rep>=n_exp_rep):
        rep_score = 3
        ang_score=3
    elif(n_rep>=n_min_rep):
        rep_score = 2
        ang_score=3

        rec = '¡Muy bien! Intenta incluir unas cuantas repeticiones más en el tiempo dado.'
        if(rec not in rep_rec):
            rep_rec.append(rec)

        rec = 'Asegúrate de tocar tu rodilla con el codo'
        if(rec not in rep_rec):
            angle_rec.append(rec)
    else:
        rep_score = 1
        ang_score=2

        rec = '¡Casi! Apura el paso, necesitas hacer al menos '+str(n_min_rep) +' repeticiones por lado'
        if(rec not in rep_rec):
            rep_rec.append(rec)

        rec = 'Asegúrate de tocar tu rodilla con el codo ¡Tú puedes!'
        if(rec not in rep_rec):
            angle_rec.append(rec)

    if(n_rep<(n_min_rep/2)):
        ang_score = 1

    if(rep_score>1):

        #Izquierda
        if(freq_max_1>(0.26*3)):
            rec = '¡Ups, parece que vas muy rápido! Intenta hacer el ejercicio un poco más lento.'
            if(rec not in recommendations):
                recommendations.append(rec)
        elif(freq_max_1<(0.26*0.5) or n_max_1==0):
            rec = '¡Vamos, vamos! Tu ritmo es un poco lento, intenta hacer el ejercicio un poco más rápido.'
            if(rec not in recommendations):
                recommendations.append(rec)
        # Derecho
        if(freq_max_2>(0.26*3)):
            rec = '¡Ups, parece que vas muy rápido! Intenta hacer el ejercicio un poco más lento.'
            if(rec not in recommendations):
                recommendations.append(rec)
        elif(freq_max_2<(0.26*0.5) or n_max_2==0):
            rec = '¡Vamos, vamos! Tu ritmo es un poco lento, intenta hacer el ejercicio un poco más rápido.'
            if(rec not in recommendations):
                recommendations.append(rec)

    #General score
    if((ang_score<2 or rep_score<2)):
        score=False

    stats={
        'original_stats':df_o_stats,
        'angle_1_stats':df_1_stats,
        'angle_2_stats':df_2_stats,
        'n_rep':int(n_rep),
        'valid_time':   None,
        'fps':          float(fps),
    }
    result={

        'angle':            ang_score,
        'rep':              rep_score,
        'ang_rec': angle_rec,
        'rep_rec': rep_rec,
        'score': score,
        'stats':stats
    }

    return result




