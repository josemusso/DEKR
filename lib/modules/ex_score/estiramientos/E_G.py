import pandas as pd
import numpy as np
import math

def score(data):
    ## Calification rules

    #Rodilla
    max_border_1 = 360     # Max threshold for valid point
    min_border_1 = 0      # Min threshold for valid point

    ang_thr=45

    time_thr_min=20
    time_thr_max=25

    # Read data
    df = data

    #Orde data
    df=df[['Second','Angle']]
    df['time']=df['Second']
    df[['ang_1']]=pd.DataFrame(df["Angle"].to_list(), columns=['ang_1'])
    df['ang_1']=df['ang_1'].astype(float)
    df.drop(columns=['Angle', 'Second'], inplace=True)

    #Fix bad frames
    df.loc[df['ang_1'] > max_border_1, 'ang_1'] = df.ang_1.mean()
    df.loc[df['ang_1'] < min_border_1, 'ang_1'] = df.ang_1.mean()

    #Calculate each frame rate
    df['frame_t'] = abs(df.time - df.time.shift(1))
    fps=round(1/df.frame_t.median(),2)

    #Save original stats
    df_o_stats=df.describe().astype(float).round(3).to_dict()

    #Get valid angles
    df_1 = df[(df.ang_1<=ang_thr)] #Select by 1st angle

    #Calculate valid exercise time
    valid_time = round(df_1.frame_t.sum(),2)

    #Save final stats
    df_stats=df_1.describe().astype(float).round(3).to_dict()


    #Evaluate
    score = True
    time_rec=[]
    angle_rec=[]

    mean_ang_o  = df.ang_1.mean()
    mean_ang    = df_1.ang_1.mean()

    """ print(mean_ang)
    print(valid_time)
    print(mean_ang_o)
    mean_ang_o=90
    valid_time=25 """

    if(valid_time<time_thr_min):
        score = False
        time_score=1
        rec='Inténtalo de nuevo, recuerda mantener la posición el máximo tiempo posible.'
        time_rec.append(rec)
        
        if(mean_ang_o>ang_thr):
            ang_score=1
            rec='Inténtalo de nuevo, recuerda que debes cruzar tu pierna sobre la otra y abrazar la pierna de arriba, llevándola hacia el hombro contrario ¡Tú puedes!'
            angle_rec.append(rec)
        else:
            ang_score=3


    elif (valid_time<time_thr_max):
        time_score=2
        rec='¡Muy bien! Lograste 20 segundos de mantención de elongación ¡Sigue así!'
        time_rec.append(rec)
        ang_score=3

    else:
        time_score=3
        ang_score=3

    stats={
        'original_stats':df_o_stats,
        'result_stats':df_stats,
        'n_rep':        None,
        'valid_time':   float(valid_time),
        'fps':          float(fps),
    }

    result={
        'time': time_score,
        'angle': ang_score,
        'ang_rec': angle_rec,
        'time_rec': time_rec,
        'score': score,
        'stats':stats
    }
    return result

