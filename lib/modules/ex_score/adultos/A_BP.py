import pandas as pd
import numpy as np
import math
from scipy.signal import argrelextrema

def score(data):
    
    score=True
    ## Calification rules

    max_border = 90             # Max threshold for valid point
    min_border = -90            # Min threshold for valid point

    # Legs
    threshold_max_1 = 25        # Threshold for valid peak 
    threshold_min_1 = -25       # Threshold for valid peak
    max_ang_1 = 35              # Threshold for great peak

    # Arms
    threshold_max_2 = 5         # Threshold for valid peak 
    threshold_min_2 = -5        # Threshold for valid peak
    max_ang_2=6                 # Threshold for great peak

    n_exp_rep = 6               # Expected repetitions
    n_min_rep=4                 # Min repetitions

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
    n=int(n)                               # Round number of frames


    # Fix unvalid points
    df.loc[df['ang_1'] > max_border, 'ang_1'] = df.ang_1.mean()
    df.loc[df['ang_1'] < min_border, 'ang_1'] = df.ang_1.mean()

    df.loc[df['ang_2'] > max_border, 'ang_2'] = df.ang_2.mean()
    df.loc[df['ang_2'] < min_border, 'ang_2'] = df.ang_2.mean()

    #Save original stats
    df_o_stats=df.describe().astype(float).round(3).to_dict()


    # Search for peaks 1
    df['min_1'] = df.iloc[argrelextrema(df.ang_1.values, np.less_equal,
                        order=n)[0]]['ang_1']
    df['max_1'] = df.iloc[argrelextrema(df.ang_1.values, np.greater_equal,
                        order=n)[0]]['ang_1']

    # Keep only points on valid threshold.
    df.loc[df['min_1'] >= threshold_min_1, 'min_1'] = np.nan
    df.loc[df['max_1'] <= threshold_max_1, 'max_1'] = np.nan


    # Search for peaks 2
    df['min_2'] = df.iloc[argrelextrema(df.ang_2.values, np.less_equal,
                        order=n)[0]]['ang_2']
    df['max_2'] = df.iloc[argrelextrema(df.ang_2.values, np.greater_equal,
                        order=n)[0]]['ang_2']

    # Keep only points on valid threshold.
    df.loc[df['min_2'] >= threshold_min_2, 'min_2'] = np.nan
    df.loc[df['max_2'] <= threshold_max_2, 'max_2'] = np.nan

    ## Cut tails.Tails it's the time the exercise hadn't started, signaled by the peaks.
    # Angle 1
    try:
        df_s = df[['time','max_1','min_1']].dropna(thresh=2)
        df_s
        start=df_s.index[0]
        end=df_s.index[-1]
        df_1 = df[start:end+1]
    except:
        pass

    # Angle 2
    try:
        df_s = df[['time','max_2','min_2']].dropna(thresh=2)
        df_s
        start=df_s.index[0]
        end=df_s.index[-1]
        df_2 = df[start:end+1]
    except:
        pass
    
    # Separate angles DF ( to clean tails)
    df_1=df_1[['time','ang_1', 'max_1','min_1']]
    df_2=df_2[['time','ang_2', 'max_2','min_2']]

    # Num of peaks 1
    n_min_1 = df_1[df_1['min_1'].notnull()].min_1.count()
    n_max_1 = df_1[df_1['max_1'].notnull()].max_1.count()

    # Peaks median 1
    med_min_1 = df_1.min_1.median()
    med_max_1 = df_1.max_1.median()

    # Num of peaks 2
    n_min_2 = df_2[df_2['min_2'].notnull()].min_2.count()
    n_max_2 = df_2[df_2['max_2'].notnull()].max_2.count()

    # Peaks median 2
    med_min_2 = df_2.min_2.median()
    med_max_2 = df_2.max_2.median()


    #Min points frequency 1
    df_min_1 = df_1[df_1['min_1'].notnull()]
    df_min_1['dif'] = abs(df_min_1.time - df_min_1.time.shift(-1))
    freq_min_1 = 1/df_min_1.dif.mean()

    #Max points frequency 1
    df_max_1 = df_1[df_1['max_1'].notnull()]
    df_max_1['dif'] = abs(df_max_1.time - df_max_1.time.shift(-1))
    freq_max_1 = 1/df_max_1.dif.mean()


    #Min points frequency 2
    df_min_2 = df_2[df_2['min_2'].notnull()]
    df_min_2['dif'] = abs(df_min_2.time - df_min_2.time.shift(-1))
    freq_min_2 = 1/df_min_2.dif.mean()

    #Max points frequency 2
    df_max_2 = df_2[df_2['max_2'].notnull()]
    df_max_2['dif'] = abs(df_max_2.time - df_max_2.time.shift(-1))
    freq_max_2 = 1/df_max_2.dif.mean()

    #Save final stats
    df_1_stats=df_1.describe().astype(float).round(3).to_dict()
    df_2_stats=df_2.describe().astype(float).round(3).to_dict()

    # Save frquency on stats
    df_1_stats['min_1']['freq']=round(freq_min_1,3)
    df_1_stats['max_1']['freq']=round(freq_max_1,3)
    df_2_stats['min_2']['freq']=round(freq_min_2,3)
    df_2_stats['max_2']['freq']=round(freq_max_2,3)

    # Results
    print('N. maximos: %d / N. minimos: %d' % (n_max_1,n_min_1))
    print('Angulo Promedio max.: %.2f / Angulo Promedio min.: %.2f' %(med_max_1,med_min_1))
    print('Freq. maximos: %.2f / Freq. minimos: %.2f' % (freq_max_1,freq_min_1))


    # Results
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

    #Repetitions
    #Arms reps
    if(n_max_1<=n_min_1):
        n_rep_1=n_max_1
    else:
        n_rep_1=n_min_1
    #Legs reps
    if(n_max_2<=n_min_2):
        n_rep_2=n_max_2
    else:
        n_rep_2=n_min_2

    #Check if difference between arms and legs it's to big
    if ((n_rep_1-n_rep_2)>2):
        rec = 'Recuerda mover tus brazos'
        if(rec not in rep_rec):
            rep_rec.append(rec)
    elif((n_rep_1-n_rep_2)<-2):
        rec = 'Recuerda mover tus piernas'
        if(rec not in rep_rec):
            rep_rec.append(rec)

    #Choose the lower between arms and legs
    if(n_rep_1<=n_rep_2):
        n_rep=n_rep_1
    else:
        n_rep=n_rep_2

    if (n_rep>=n_exp_rep):
        rep_score = 3
    elif(n_rep>=n_min_rep):
        rep_score = 2
        rec = '¡Muy bien! Intenta incluir unas cuantas repeticiones más en el tiempo dado.'
        if(rec not in rep_rec):
            rep_rec.append(rec)
    else:
        rep_score = 1
        rec = '¡Casi! Apura el paso, necesitas hacer al menos '+str(n_min_rep)+' repeticiones por lado ¡Tú puedes!'
        if(rec not in rep_rec):
            rep_rec.append(rec)

    #Pierna derecha
    if(n_max_1<(n_min_rep/2)):
        ang_pd_score = 1
        rec = 'Inténtalo de nuevo, debes mover más tu pierna derecha manteniendo la izquierdo hacia el centro de tu cuerpo ¡Tú puedes!'
        if(rec not in angle_rec):
            angle_rec.append(rec)
    elif(med_max_1>=max_ang_1):
        ang_pd_score = 3
    else:
        ang_pd_score = 2
        rec = '¡Vas por muy buen camino! Intenta mover un poco más tu pierna derecha manteniendo la izquierda hacia el centro de tu cuerpo'
        if(rec not in angle_rec):
            angle_rec.append(rec)


    #Pierna izquierda
    if(n_min_1<(n_min_rep/2)):
        ang_pi_score = 1
        rec = 'Inténtalo de nuevo, debes mover más tu pierna derecha manteniendo la izquierdo hacia el centro de tu cuerpo ¡Tú puedes!'
        if(rec not in angle_rec):
            angle_rec.append(rec)
    elif(med_min_1<=(-max_ang_1)):
        ang_pi_score = 3
    else:
        ang_pi_score = 2
        rec = '¡Vas por muy buen camino! Intenta mover un poco más tu pierna derecha manteniendo la izquierda hacia el centro de tu cuerpo'
        if(rec not in angle_rec):
            angle_rec.append(rec)

    if(rep_score>1):
        if((freq_min_1>(0.333*2)) or (freq_max_1>(0.333*2))):
            rec = '¡Ups, parece que vas muy rápido! Intenta hacer el ejercicio un poco más lento'
            recommendations.append(rec)
        elif((freq_min_1<(0.333*0.5)) or (freq_max_1<(0.333*0.5)) or n_max_1==0 or n_min_1==0):
            rec = '¡Vamos, vamos! Tu ritmo es un poco lento, intenta hacer el ejercicio un poco más rápido.'
            recommendations.append(rec)


    #Brazo derecho
    if(n_max_2<(n_min_rep/2)):
        ang_bd_score = 1
        rec = 'Inténtalo de nuevo, debes mover más tu brazo derecho manteniendo el izquierdo pegado a tu cuerpo  ¡Tú puedes!'
        if(rec not in angle_rec):
            angle_rec.append(rec)
    elif(med_max_2>=max_ang_2):
        ang_bd_score = 3
    else:
        ang_bd_score = 2
        rec = '¡Vas por muy buen camino! Intenta mover un poco más tu brazo derecho manteniendo el izquierdo pegado a tu cuerpo'
        if(rec not in angle_rec):
            angle_rec.append(rec)


    #Brazo izquierdo
    if(n_min_2<(n_min_rep/2)):
        ang_bi_score = 1
        rec = 'Inténtalo de nuevo, debes mover más tu brazo izquierdo manteniendo el derecho pegado a tu cuerpo  ¡Tú puedes!'
        if(rec not in angle_rec):
            angle_rec.append(rec)
    elif(med_min_2<=(-max_ang_2)):
        ang_bi_score = 3
    else:
        ang_bi_score = 2
        rec = '¡Vas por muy buen camino! Intenta mover un poco más tu brazo izquierdo manteniendo el derecho pegado a tu cuerpo'
        if(rec not in angle_rec):
            angle_rec.append(rec)

    if((freq_min_2>(0.333*2)) or (freq_max_2>(0.333*2))):
        rec = '¡Ups, parece que vas muy rápido! Intenta hacer el ejercicio un poco más lento'
        if(rec not in recommendations):
            recommendations.append(rec)
    elif((freq_min_2<(0.333*0.5)) or (freq_max_2<(0.333*0.5)) or n_max_2==0 or n_min_2==0):
        rec = '¡Vamos, vamos! Tu ritmo es un poco lento, intenta hacer el ejercicio un poco más rápido.'
        if(rec not in recommendations):
            recommendations.append(rec)


    #Angle score
    if(ang_pi_score<2 or ang_pd_score<2 or 
        ang_bi_score<2 or ang_bd_score<2):
        ang_score=1
    elif(ang_pi_score<3 or ang_pd_score<3 or 
        ang_bi_score<3 or ang_bd_score<3):
        ang_score=2
    else:
        ang_score=3

    if(len(angle_rec)>=3):
        angle_rec=[]
        if(ang_pi_score<2 or ang_pd_score<2):
            rec = 'Inténtalo de nuevo, debes mover más tu pierna manteniendo la contraria hacia el centro de tu cuerpo ¡Tú puedes!'
            angle_rec.append(rec)
        else:
            rec = '¡Vas por muy buen camino! Intenta mover un poco más tu pierna manteniendo la contraria hacia el centro de tu cuerpo'
            angle_rec.append(rec)

        if(ang_bi_score<2 or ang_bd_score<2):
            rec = 'Inténtalo de nuevo, debes elevar más tu brazo manteniendo el contrario pegado a tu cuerpo  ¡Tú puedes!'
            angle_rec.append(rec)
        else:
            rec = 'Vas por muy buen camino! Intenta mover un poco más tu brazo manteniendo el contrario pegado a tu cuerpo'
            angle_rec.append(rec)

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

