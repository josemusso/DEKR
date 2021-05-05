import pandas as pd
import numpy as np
import math
from scipy.signal import argrelextrema


def score(data):

    score = True
    ## Calification rules
    max_border = 100         # Max threshold for valid point
    min_border = -100        # Min threshold for valid point

    min_ang = 10                    # Threshold for valid max
    mid_ang = 15                    # Threshold for great max

    n_exp_rep = 40                   # Number of desired reps
    n_min_rep = 30                   # Number of minimum reps

    #Read data
    df = data

    # Order data
    df=df[['Second','Angle']]
    df[['ang_1']]=pd.DataFrame(df["Angle"].to_list(), columns=['ang_1'])
    df['time']=df['Second']
    df.drop(columns=['Angle', 'Second'], inplace=True)

    df['ang_1']=df['ang_1'].astype(float)

    # Calculate aprox frame rate and set number of frames to compare
    max_window=0.5                                            # Time window (seconds)
    df['frame_t'] = abs(df.time - df.time.shift(1))         # Get time difference for every frame
    mean_ftime=df.frame_t.median()                          # Get average difference
    fps=round(1/mean_ftime,2)                               # Aproximate FPS
    n=max_window/mean_ftime                                 # Number of frames in the time window
    n=int(n)                               # Round number of frames

    # Fix unvalid points
    df.loc[df['ang_1'] > max_border, 'ang_1'] = df.ang_1.mean()
    df.loc[df['ang_1'] < min_border, 'ang_1'] = df.ang_1.mean()

    #Save original stats
    df_o_stats=df.describe().astype(float).round(3).to_dict()


    # Search for max and min points
    df['min_1'] = df.iloc[argrelextrema(df.ang_1.values, np.less_equal,
                        order=n)[0]]['ang_1']
    df['max_1'] = df.iloc[argrelextrema(df.ang_1.values, np.greater_equal,
                        order=n)[0]]['ang_1']

    # Keep only points on valid threshold.
    df.loc[df['max_1'] <= min_ang, 'max_1'] = np.nan


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

    # Separate angles DF ( to clean tails)
    df_1=df_1[['time','ang_1', 'max_1','min_1']]

    ## Cut tails
    try:
        # Angle 1
        df_s = df_1[df_1['max_1'].notnull()].copy()
        start=df_s.index[0]
        end=df_s.index[-1]
        df_1 = df_1[start:end+1]
    except:
        pass

    # Number of peaks.
    n_min_1 = df_1[df_1['min_1'].notnull()].min_1.count()
    n_max_1 = df_1[df_1['max_1'].notnull()].max_1.count()

    # Peaks median.
    med_min_1 = df_1.min_1.median()
    med_max_1 = df_1.max_1.median()

    #Min points frequency
    df_min_1 = df[df['min_1'].notnull()]
    df_min_1['dif'] = abs(df_min_1.time - df_min_1.time.shift(-1))
    freq_min_1 = 1/df_min_1.dif.mean()

    #Max points frequency
    df_max_1 = df[df['max_1'].notnull()]
    df_max_1['dif'] = abs(df_max_1.time - df_max_1.time.shift(-1))
    freq_max_1 = 1/df_max_1.dif.mean()

    #Save final stats
    df_1_stats=df_1.describe().astype(float).round(3).to_dict()

    # Save frquency on stats
    df_1_stats['min_1']['freq']=round(freq_min_1,3)
    df_1_stats['max_1']['freq']=round(freq_max_1,3)

    # Resultados Izquierda.
    print('N. maximos: %d / N. minimos: %d' % (n_max_1,n_min_1))
    print('Angulo Promedio max.: %.2f / Angulo Promedio min.: %.2f' %(med_max_1,med_min_1))
    print('Freq. maximos: %.2f / Freq. minimos: %.2f' % (freq_max_1,freq_min_1))

    score=True
    """ n_max_1=30
    med_max_1=10 """

    rep_rec=[]
    angle_rec=[]
    recommendations=[]

    #Repetitions
    n_rep=n_max_1
    if (n_rep>=n_exp_rep):
        rep_score = 3
    elif(n_rep>=n_min_rep):
        rep_score = 2
        rec = '¡Muy bien! Intenta incluir unas cuantas repeticiones más en el tiempo dado.'
        if(rec not in rep_rec):
            rep_rec.append(rec)
    else:
        rep_score = 1
        rec = '¡Casi! Apura el paso, necesitas hacer al menos '+str(n_min_rep) +' repeticiones por lado ¡Tú puedes!'
        if(rec not in rep_rec):
            rep_rec.append(rec)

    #Extension
    if(n_rep<(n_min_rep/2)):
        ang_score = 1
        rec = 'Inténtalo de nuevo, debes marcar mejor los pasos ¡Tú puedes!'
        if(rec not in angle_rec):
            angle_rec.append(rec)
    elif(med_max_1>=mid_ang):
        ang_score = 3
    else:
        ang_score = 2
        rec = '¡Vas por muy buen camino! Intenta hacer un poco más marcados los pasos '
        if(rec not in angle_rec):
            angle_rec.append(rec)

    if(rep_score>1):
        if(freq_max_1>(0.66*2)):
            rec = '¡Ups, parece que vas muy rápido! Intenta hacer el ejercicio un poco más lento.'
            recommendations.append(rec)
        elif(freq_max_1<(0.66*0.5) or n_max_1==0):
            rec = '¡Vamos, vamos! Tu ritmo es un poco lento, intenta hacer el ejercicio un poco más rápido.'
            recommendations.append(rec)

    #General score
    if((ang_score<2 or rep_score<2)):
        score=False


    stats={
        'original_stats':   df_o_stats,
        'angle_1_stats':    df_1_stats,
        'n_rep':int(n_rep),
        'valid_time':   None,
        'fps':          float(fps),
    }
    result={

        'angle':            ang_score,
        'rep':              rep_score,
        'ang_rec': angle_rec,
        'rep_rec': rep_rec,
        'score':            score,
        'stats':            stats
    }

    return result




