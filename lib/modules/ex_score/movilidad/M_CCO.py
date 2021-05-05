import pandas as pd
import numpy as np
import math
from scipy.signal import argrelextrema

def score(data):
    ## Calification rules

    max_border = 360        # Max threshold for valid point
    min_border = 0          # Min threshold for valid point

    base_ang = 180          # Objective angle
    threshold_ang = 90      # Threshold for valid maxium or minimum

    n_exp_rep = 12          # Number of expected reps
    n_min_rep = 8            # Number of minimum reps


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
    df.loc[df['ang_1'] > max_border, 'ang_1'] = df.ang_1.mean()
    df.loc[df['ang_1'] < min_border, 'ang_1'] = df.ang_1.mean()

    df.loc[df['ang_2'] > max_border, 'ang_2'] = df.ang_2.mean()
    df.loc[df['ang_2'] < min_border, 'ang_2'] = df.ang_2.mean()

    #Save original stats
    df_o_stats=df.describe().astype(float).round(3).to_dict()


    # Search for max and min points - angle 1
    df['min_1'] = df.iloc[argrelextrema(df.ang_1.values, np.less_equal,
                        order=n)[0]]['ang_1']
    df['max_1'] = df.iloc[argrelextrema(df.ang_1.values, np.greater_equal,
                        order=n)[0]]['ang_1']

    # Keep only points on valid threshold.
    df.loc[df['min_1'] >= base_ang-threshold_ang, 'min_1'] = np.nan
    df.loc[df['max_1'] <= base_ang+threshold_ang, 'max_1'] = np.nan


    # Search for max and min points - angle 2
    df['min_2'] = df.iloc[argrelextrema(df.ang_2.values, np.less_equal,
                        order=n)[0]]['ang_2']
    df['max_2'] = df.iloc[argrelextrema(df.ang_2.values, np.greater_equal,
                        order=n)[0]]['ang_2']

    # Keep only points on valid threshold.
    df.loc[df['min_2'] >= base_ang-threshold_ang, 'min_2'] = np.nan
    df.loc[df['max_2'] <= base_ang+threshold_ang, 'max_2'] = np.nan

    # Separate angles DF ( to clean tails)
    df_1=df[['time','ang_1', 'max_1','min_1']]
    df_2=df[['time','ang_2', 'max_2','min_2']]

    ## Cut tails.Tails it's the time the exercise hadn't started, signaled by the peaks.
    # Angle 1
    try:
        df_s = df_1[['time','max_1','min_1']].dropna(thresh=2)
        df_s
        start=df_s.index[0]
        end=df_s.index[-1]
        df_2 = df_2[start:end+1]
    except:
        pass
    # Angle 2
    try:
        df_s = df_2[['time','max_2','min_2']].dropna(thresh=2)
        df_s
        start=df_s.index[0]
        end=df_s.index[-1]
        df_2 = df_2[start:end+1]
    except:
        pass

    # Num of peaks - angle 1
    n_min_1 = df_1[df_1['min_1'].notnull()].min_1.count()
    n_max_1 = df_1[df_1['max_1'].notnull()].max_1.count()

    # Peaks median - 1
    med_min_1 = df_1.min_1.median()
    med_max_1 = df_1.max_1.median()

    # Num of peaks - angle 2
    n_min_2 = df_2[df_2['min_2'].notnull()].min_2.count()
    n_max_2 = df_2[df_2['max_2'].notnull()].max_2.count()

    # Peaks median - angle 2
    med_min_2 = df_2.min_2.median()
    med_max_2 = df_2.max_2.median()

    #Save final stats
    df_1_stats=df_1.describe().astype(float).round(3).to_dict()
    df_2_stats=df_2.describe().astype(float).round(3).to_dict()


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

    # Save frquency on stats
    df_1_stats['min_1']['freq']=round(freq_min_1,3)
    df_1_stats['max_1']['freq']=round(freq_max_1,3)
    df_2_stats['min_2']['freq']=round(freq_min_2,3)
    df_2_stats['max_2']['freq']=round(freq_max_2,3)


    # Results 1.
    print('N. maximos: %d / N. minimos: %d' % (n_max_1,n_min_1))
    print('Angulo Promedio max.: %.2f / Angulo Promedio min.: %.2f' %(med_max_1,med_min_1))
    print('Freq. maximos: %.2f / Freq. minimos: %.2f' % (freq_max_1,freq_min_1))
    # Results 2.
    print('N. maximos: %d / N. minimos: %d' % (n_max_2,n_min_2))
    print('Angulo Promedio max.: %.2f / Angulo Promedio min.: %.2f' %(med_max_2,med_min_2))
    print('Freq. maximos: %.2f / Freq. minimos: %.2f' % (freq_max_2,freq_min_2))

    
    score = True

    """ n_max_1=4
    n_min_1=4
    n_max_2=4
    n_min_2=4
    med_max_1=25
    med_min_1=-25
    med_max_2=5
    med_min_2=-5 """

    rep_rec=[]
    angle_rec=[]
    recommendations=[]
    #Brazo derecho
    if(n_max_1<n_min_1):
        n_rep_der = n_min_1
    else:
        n_rep_der = n_max_1

    if (n_rep_der>=n_exp_rep):
        rep_d_score = 3
        ang_d_score = 3
    elif(n_rep_der>=n_min_rep):
        rep_d_score = 2
        ang_d_score = 3

        rec = '¡Muy bien! Intenta incluir unas cuantas repeticiones más en el tiempo dado.'
        if(rec not in rep_rec):
            rep_rec.append(rec)

        rec = 'Recuerda realizar circulos amplios manteniendo tus codos cerca de tu cuerpo.'
        if(rec not in rep_rec):
            angle_rec.append(rec)
    else:
        rep_d_score = 1
        ang_d_score = 2

        rec = '¡Casi! Apura el paso, necesitas hacer al menos '+str(n_min_rep)+' repeticiones.'
        if(rec not in rep_rec):
            rep_rec.append(rec)

        rec = 'Recuerda realizar circulos amplios manteniendo tus codos cerca de tu cuerpo ¡Tú puedes!'
        if(rec not in rep_rec):
            angle_rec.append(rec)

    if(n_rep_der<(n_min_rep/2)):
        ang_d_score = 1

    #Brazo izquierdo
    if(n_max_2<n_min_2):
        n_rep_izq = n_min_2
    else:
        n_rep_izq = n_max_2

    if (n_rep_izq>=n_exp_rep):
        rep_i_score = 3
        ang_i_score = 3
    elif(n_rep_izq>=n_min_rep):
        rep_i_score = 2
        ang_i_score = 3
        rec = '¡Muy bien! Intenta incluir unas cuantas repeticiones más en el tiempo dado.'
        if(rec not in rep_rec):
            rep_rec.append(rec)

        rec = 'Recuerda realizar circulos amplios manteniendo tus codos cerca de tu cuerpo.'
        if(rec not in rep_rec):
            angle_rec.append(rec)

    else:
        rep_i_score = 1
        ang_i_score = 2
        rec = '¡Casi! Apura el paso, necesitas hacer al menos '+str(n_min_rep)+' repeticiones.'
        if(rec not in rep_rec):
            rep_rec.append(rec)

        rec = 'Recuerda realizar circulos amplios manteniendo tus codos cerca de tu cuerpo ¡Tú puedes!'
        if(rec not in rep_rec):
            angle_rec.append(rec)

    if(n_rep_izq<(n_min_rep/2)):
        ang_i_score = 1

    #Angle score
    if(ang_d_score<ang_i_score):
        ang_score=ang_d_score
    else:
        ang_score=ang_i_score

    #Repetitions score
    if(rep_d_score<rep_i_score):
        rep_score=rep_d_score
        n_rep = n_rep_der
    else:
        rep_score=rep_i_score
        n_rep = n_rep_izq

    if(rep_score>1):
        if((freq_min_1>(0.5*2)) or (freq_max_1>(0.5*2))):
            rec = '¡Ups, parece que vas muy rápido! Intenta hacer el ejercicio un poco más lento'
            recommendations.append(rec)
        elif((freq_min_1<(0.5*0.5)) or (freq_max_1<(0.5*0.5)) or n_max_1==0 or n_min_1==0):
            rec = '¡Vamos, vamos! Tu ritmo es un poco lento, intenta hacer el ejercicio un poco más rápido.'
            recommendations.append(rec)

        if((freq_min_2>(0.5*2)) or (freq_max_2>(0.5*2))):
            rec = '¡Ups, parece que vas muy rápido! Intenta hacer el ejercicio un poco más lento'
            if(rec not in recommendations):
                recommendations.append(rec)
        elif((freq_min_2<(0.5*0.5)) or (freq_max_2<(0.5*0.5)) or n_max_2==0 or n_min_2==0):
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






