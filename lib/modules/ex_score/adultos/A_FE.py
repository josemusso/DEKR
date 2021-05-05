import pandas as pd
import numpy as np
import math
from scipy.signal import argrelextrema


#1. Arms
#2. Legs

def score(data):

    ## Calification rules

    # Arm
    max_border_1 = 250              # Max threshold for valid point
    min_border_1 = 0                # Min threshold for valid point

    min_threshold_1 = 40            # Threshold for min peak 
    min_ang_1 = 100                 # Threshold for valid peak 
    mid_ang_1 = 120                 # Threshold for great peak 

    # Leg
    max_border_2 = 300              # Max threshold for valid point
    min_border_2 = 100              # Min threshold for valid point

    min_threshold_2 = 180           # Threshold for min peak 
    min_ang_2 = 185                 # Threshold for valid peak
    mid_ang_2 = 200                 # Threshold for great peak 

    n_exp_rep = 8                   # Expected repetitions
    n_min_rep = 6                   # Min repetitions


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

    # Save original stats
    df_o_stats=df.describe().astype(float).round(3).to_dict()


    # Search for max and min points
    df['min_1'] = df.iloc[argrelextrema(df.ang_1.values, np.less_equal,
                        order=n)[0]]['ang_1']
    df['max_1'] = df.iloc[argrelextrema(df.ang_1.values, np.greater_equal,
                        order=n)[0]]['ang_1']

    # Keep only points on valid threshold.
    df.loc[df['min_1'] >= min_threshold_1, 'min_1'] = np.nan
    df.loc[df['max_1'] <= min_ang_1, 'max_1'] = np.nan


    # Search for max and min points
    df['min_2'] = df.iloc[argrelextrema(df.ang_2.values, np.less_equal,
                        order=n)[0]]['ang_2']
    df['max_2'] = df.iloc[argrelextrema(df.ang_2.values, np.greater_equal,
                        order=n)[0]]['ang_2']

    # Keep only points on valid threshold.
    df.loc[df['min_2'] >= min_threshold_2, 'min_2'] = np.nan
    df.loc[df['max_2'] <= min_ang_2, 'max_2'] = np.nan


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
        df_s = df_1[df_1['max_1'].notnull()].copy()
        start=df_s.index[0]
        end=df_s.index[-1]
        df_1 = df_1[start:end+1]
    except:
        pass
    try:
        # Angle 2
        df_s = df_2[df_2['max_2'].notnull()].copy()
        start=df_s.index[0]
        end=df_s.index[-1]
        df_2 = df_2[start:end+1]
    except:
        pass

    # Num of peaks 1
    n_min_1 = df_1[df_1['min_1'].notnull()].min_1.count()
    n_max_1 = df_1[df_1['max_1'].notnull()].max_1.count()

    # peaks median 1
    med_min_1 = df_1.min_1.median()
    med_max_1 = df_1.max_1.median()

    # Num of peaks 2
    n_min_2 = df_2[df_2['min_2'].notnull()].min_2.count()
    n_max_2 = df_2[df_2['max_2'].notnull()].max_2.count()

    # Peaks median 2
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

    #Save final stats
    df_1_stats=df_1.describe().astype(float).round(3).to_dict()
    df_2_stats=df_2.describe().astype(float).round(3).to_dict()

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


    score=True
    """ med_max_1=150
    med_max_2=170 """

    rep_rec=[]
    angle_rec=[]
    recommendations=[]

    # Repetitions

    #Check if difference between arms and legs it's to big
    if ((n_max_1-n_max_2)>2):
        rec = '¡Intenta mover más tus piernas!'
        if(rec not in rep_rec):
            rep_rec.append(rec)
    elif((n_max_1-n_max_2)<-2):
        rec = '¡No olvides mover los brazos!'
        if(rec not in rep_rec):
            rep_rec.append(rec)

    if(n_max_1<=n_max_2):
        n_rep=n_max_1
    else:
        n_rep=n_max_2
        
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

    # Brazo
    if(n_max_1<(n_min_rep/2)):
        ang_score_1 = 1
        rec = 'Inténtalo de nuevo, debes levantar más tus brazos ¡Tú puedes!'
        if(rec not in recommendations):
            recommendations.append(rec)
    elif(med_max_1>=mid_ang_1):
        ang_score_1 = 3
    else:
        ang_score_1 = 2
        rec = '¡Vas por muy buen camino! Intenta elevar un poco más tus brazos '
        if(rec not in recommendations):
            recommendations.append(rec)

    if(freq_max_1>(0.26*2)):
        rec = '¡Ups, parece que vas muy rápido! Intenta hacer el ejercicio un poco más lento.'
        if(rec not in recommendations):
            recommendations.append(rec)
    elif(freq_max_1<(0.26*0.5) or n_max_1==0):
        rec = '¡Vamos, vamos! Tu ritmo es un poco lento, intenta hacer el ejercicio un poco más rápido.'
        if(rec not in recommendations):
            recommendations.append(rec)

    # Pierna
    if(n_max_2<(n_min_rep/2)):
        ang_score_2 = 1
        rec = 'Inténtalo de nuevo, debes llevar tu pierna extendida más hacia atrás ¡Tú puedes!'
        if(rec not in recommendations):
            recommendations.append(rec)
    elif(med_max_2>=mid_ang_2):
        ang_score_2 = 3
    else:
        ang_score_2 = 2
        rec = '¡Vas por muy buen camino! Intenta llevar tu pierna extendida un poco más hacia atrás'
        if(rec not in recommendations):
            recommendations.append(rec)

    if(rep_score>1):
        if(freq_max_2>(0.26*2)):
            rec = '¡Ups, parece que vas muy rápido! Intenta hacer el ejercicio un poco más lento.'
            if(rec not in recommendations):
                recommendations.append(rec)
        elif(freq_max_2<(0.26*0.5) or n_max_2==0):
            rec = '¡Vamos, vamos! Tu ritmo es un poco lento, intenta hacer el ejercicio un poco más rápido.'
            if(rec not in recommendations):
                recommendations.append(rec)

    #Angle score
    if(ang_score_1<ang_score_2):
        ang_score=ang_score_1
    else:
        ang_score=ang_score_2

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


