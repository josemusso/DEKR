import pandas as pd
import numpy as np
import math

def score(data):
    ## Calification rules
    max_border = 30     # Max threshold for valid point
    min_border = 0      # Min threshold for valid point

    min_thr=10
    mid_thr=20

    time_thr_min = 20       # Minimun valid time
    time_thr_max = 25

    # Read data
    df = data

    #Orde data
    df=df[['Second','Angle']]
    df['time']=df['Second']
    df[['ang']]=pd.DataFrame(df["Angle"].to_list(), columns=['ang'])
    df['ang']=df['ang'].astype(float)
    df.drop(columns=['Angle', 'Second'], inplace=True)

    #Fix bad frames
    df.loc[df['ang'] > max_border, 'ang'] = df.ang.mean()
    df.loc[df['ang'] < min_border, 'ang'] = df.ang.mean()

    #Calculate each frame rate
    df['frame_t'] = abs(df.time - df.time.shift(1))
    fps=round(1/df.frame_t.median(),2)

    #Save original stats
    df_o_stats=df.describe().astype(float).round(3).to_dict()

    mean_ang_o = df.ang.mean()
    std = df.ang.std()  #68.2

    #Get valid angles
    df_2 = df[(df.ang>=min_thr)]
    mean_ang=df_2.ang.mean()

    #Save final stats
    df_stats=df_2.describe().astype(float).round(3).to_dict()

    #Calculate valid exercise time
    valid_time = round(df_2.frame_t.sum(),2)

    #Evaluate
    score = True
    time_rec=[]
    angle_rec=[]
    """ print(mean_ang)
    print(valid_time)
    print(mean_ang_o)
    mean_ang=20
    valid_time=25 """

    if(valid_time<time_thr_min):
        score = False
        time_score=1
        rec='Inténtalo de nuevo, recuerda mantener la posición el máximo tiempo posible.'
        time_rec.append(rec)

        if(mean_ang_o<min_thr):
            ang_score=1
            rec='Busca inclinar más la cabeza hacia el costado ¡Tú puedes!'
            angle_rec.append(rec)
        elif(mean_ang_o<mid_thr):
            ang_score=2
            rec='¡Vas por muy buen camino! Intenta inclinar aún más la cabeza.'
            angle_rec.append(rec)
        else:
            ang_score=3

    elif (valid_time<time_thr_max):
        time_score=2
        rec='¡Muy bien! Lograste 20 segundos de mantención de elongación ¡Sigue así!'
        time_rec.append(rec)
        
        if(mean_ang<mid_thr):
            ang_score=2
            rec='¡Vas por muy buen camino! Intenta inclinar aún más la cabeza.'
            angle_rec.append(rec)
        else:
            ang_score=3
    else:
        time_score=3

        if(mean_ang<mid_thr):
            ang_score=2
            rec='¡Vas por muy buen camino! Intenta inclinar aún más la cabeza.'
            angle_rec.append(rec)
        else:
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
