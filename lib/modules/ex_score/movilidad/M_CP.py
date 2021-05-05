import pandas as pd
import numpy as np
import math
from scipy.signal import argrelextrema

def score(data):
    ## Calification rules

    max_border = 120        # Max threshold for valid point
    min_border = -20        # Min threshold for valid point

    base_ang = 60           # Objective angle
    threshold_ang = 10      # Threshold for valid peak
    max_ang = 15            # Threshold for great peak

    n_exp_rep = 5           # Expected repetitions
    n_min_rep= 4            # minimum reps

    #Read data
    df = data

    # Order data
    df=df[['Second','Angle']]
    df[['ang_1']]=pd.DataFrame(df["Angle"].to_list(), columns=['ang_1'])
    df['time']=df['Second']
    df.drop(columns=['Angle', 'Second'], inplace=True)
    df['ang_1']=df['ang_1'].astype(float)

    # Calculate aprox frame rate and set number of frames to compare
    max_window=1                                            # Time window (seconds)
    df['frame_t'] = abs(df.time - df.time.shift(1))         # Get time difference for every frame
    mean_ftime=df.frame_t.median()                          # Get average difference
    fps=round(1/mean_ftime,2)                               # Aproximate FPS
    n=max_window/mean_ftime                                 # Number of frames in the time window
    n=int(n)                                # Round number of frames

    # Fix unvalid points
    df.loc[df['ang_1'] > max_border, 'ang_1'] = df.ang_1.mean()
    df.loc[df['ang_1'] < min_border, 'ang_1'] = df.ang_1.mean()

    #Save original stats
    df_o_stats=df.describe().astype(float).round(3).to_dict()

    # Search for peaks
    df['min_1'] = df.iloc[argrelextrema(df.ang_1.values, np.less_equal,
                        order=n)[0]]['ang_1']
    df['max_1'] = df.iloc[argrelextrema(df.ang_1.values, np.greater_equal,
                        order=n)[0]]['ang_1']

    # Keep only points on valid threshold.
    df.loc[df['min_1'] >= base_ang-threshold_ang, 'min_1'] = np.nan
    df.loc[df['max_1'] <= base_ang+threshold_ang, 'max_1'] = np.nan

    # Num of peaks.
    n_min_1 = df[df['min_1'].notnull()].min_1.count()
    n_max_1 = df[df['max_1'].notnull()].max_1.count()

    ## Cut tails.Tails it's the time the exercise hadn't started, signaled by the peaks.
    # Angle 1
    try:
        df_s = df[['time','max_1','min_1']].dropna(thresh=2)
        df_s
        start=df_s.index[0]
        end=df_s.index[-1]
        df = df[start:end+1]
    except:
        pass

    #Save final stats
    df_stats=df.describe().astype(float).round(3).to_dict()

    # Peaks median.
    med_min_1 = df.min_1.median()
    med_max_1 = df.max_1.median()

    #Min points frequency
    df_min_1 = df[df['min_1'].notnull()]
    df_min_1['dif'] = abs(df_min_1.time - df_min_1.time.shift(-1))
    freq_min_1 = 1/df_min_1.dif.mean()

    #Max points frequency
    df_max_1 = df[df['max_1'].notnull()]
    df_max_1['dif'] = abs(df_max_1.time - df_max_1.time.shift(-1))
    freq_max_1 = 1/df_max_1.dif.mean()

    #Save frquency
    df_stats['min_1']['freq']=round(freq_min_1,3)
    df_stats['max_1']['freq']=round(freq_max_1,3)


    # Results
    print('N. maximos: %d / N. minimos: %d' % (n_max_1,n_min_1))
    print('Angulo Promedio max.: %.2f / Angulo Promedio min.: %.2f' %(med_max_1,med_min_1))
    print('Freq. maximos: %.2f / Freq. minimos: %.2f' % (freq_max_1,freq_min_1))

    score = True

    """ n_max_1=4
    n_min_1=4
    med_max_1=70
    med_min_1=50 """

    rep_rec=[]
    angle_rec=[]
    recommendations=[]

    # Repetitions
    if(n_max_1<=n_min_1):
        n_rep=n_max_1
    else:
        n_rep=n_min_1
        
    if (n_rep>=n_exp_rep):
        rep_score = 3
    elif(n_rep>=n_min_rep):
        rep_score = 2
        rec = '¡Muy bien! Intenta incluir unas cuantas repeticiones más en el tiempo dado.'
        if(rec not in rep_rec):
            rep_rec.append(rec)
    else:
        rep_score = 1
        rec = '¡Casi! Apura el paso, necesitas hacer al menos '+str(n_min_rep)+' repeticiones ¡Tú puedes!'
        if(rec not in rep_rec):
            rep_rec.append(rec)

    #Extension
    if(n_max_1<(n_min_rep/2)):
        ang_1_score = 1
        rec = 'Inténtalo de nuevo, debes inclinar tu cabeza más hacia atrás ¡Tú puedes!'
        if(rec not in angle_rec):
            angle_rec.append(rec)
    elif(med_max_1>=base_ang+max_ang):
        ang_1_score = 3
    else:
        ang_1_score = 2
        rec = '¡Vas por muy buen camino! Intenta inclinar tu cabeza un poco más hacia atrás'
        if(rec not in angle_rec):
            angle_rec.append(rec)


    #Flexion
    if(n_min_1<(n_min_rep/2)):
        ang_2_score = 1
        rec = 'Inténtalo de nuevo, debes doblar tu cabeza más hacia adelante ¡Tú puedes!'
        if(rec not in angle_rec):
            angle_rec.append(rec)
    elif(med_min_1<=(base_ang-max_ang)):
        ang_2_score = 3
    else:
        ang_2_score = 2
        rec = '¡Vas por muy buen camino! Intenta doblar tu cabeza un poco más hacia adelante'
        if(rec not in angle_rec):
            angle_rec.append(rec)

    if(rep_score>1):
        if((freq_min_1>(0.1666*2)) or (freq_max_1>(0.1666*2))):
            rec = '¡Ups, parece que vas muy rápido! Intenta hacer el ejercicio un poco más lento'
            recommendations.append(rec)
        elif((freq_min_1<(0.1666*0.5)) or (freq_max_1<(0.1666*0.5))):
            rec = '¡Vamos, vamos! Tu ritmo es un poco lento, intenta hacer el ejercicio un poco más rápido.'
            recommendations.append(rec)


    #Angle score
    if(ang_1_score<ang_2_score):
        ang_score=ang_1_score
    else:
        ang_score=ang_2_score


    #General score
    if((ang_score<2 or rep_score<2)):
        score=False

    stats={
        'original_stats':df_o_stats,
        'angle_1_stats':df_stats,
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

