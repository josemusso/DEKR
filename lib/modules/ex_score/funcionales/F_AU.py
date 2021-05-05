import pandas as pd
import numpy as np
import math

def score(data):

    score=True
    ## Calification rules

    # Rodilla
    max_border_1 = 210     # Max threshold for valid point
    min_border_1 = 0      # Min threshold for valid point

    min_thr_1 = 120
    mid_thr_1 = 110
    
    # Cadera
    max_border_2 = 210     # Max threshold for valid point
    min_border_2 = 0      # Min threshold for valid point
    
    min_thr_2 = 120
    mid_thr_2 = 100
    
    time_thr_min = 15       # Minimun valid time
    time_thr_max = 20

    # Read data
    df = data

    #Orde data
    df=df[['Second','Angle']]
    df['time']=df['Second']
    df[['ang_1','ang_2']]=pd.DataFrame(df["Angle"].to_list(), columns=['ang_1','ang_2'])
    df['ang_1']=df['ang_1'].astype(float)
    df['ang_2']=df['ang_2'].astype(float)
    df.drop(columns=['Angle', 'Second'], inplace=True)

    #Fix bad frames
    df.loc[df['ang_1'] > max_border_1, 'ang_1'] = df.ang_1.mean()
    df.loc[df['ang_1'] < min_border_1, 'ang_1'] = df.ang_1.mean()

    df.loc[df['ang_2'] > max_border_2, 'ang_2'] = df.ang_2.mean()
    df.loc[df['ang_2'] < min_border_2, 'ang_2'] = df.ang_2.mean()

    #Calculate each frame rate
    df['frame_t'] = abs(df.time - df.time.shift(1))
    fps=round(1/df.frame_t.median(),2)

    #Save original stats
    df_o_stats=df.describe().astype(float).round(3).to_dict()

    #Get valid angles
    df_1 = df[(df.ang_1<=min_thr_1)] #Select by 1st angle

    df_2 = df_1[(df_1.ang_2<=min_thr_2)] #Select by 2nd angle

    valid_time = df_2.frame_t.sum()
    valid_time = round(valid_time,2)

    #Save final stats
    df_stats=df_2.describe().astype(float).round(3).to_dict()

    #Evaluate
    score = True
    time_rec=[]
    angle_rec=[]

    mean_ang_1_o=df.ang_1.mean()
    mean_ang_2_o=df.ang_2.mean()

    mean_ang_1=df_2.ang_1.mean()
    mean_ang_2=df_2.ang_2.mean()

    """ mean_ang_1=60
    mean_ang_2=110
    valid_time=18 """

    print(mean_ang_1)
    print(mean_ang_2)
    print(valid_time)

    if(valid_time<time_thr_min):
        score = False
        time_score=1
        rec='Inténtalo de nuevo, recuerda mantener la posición el máximo tiempo posible.'
        time_rec.append(rec)

        #Cadera
        if(mean_ang_2_o<=mid_thr_2):
            ang_2_score=3
        elif(mean_ang_2_o<=min_thr_2):
            ang_2_score=2
            rec='¡Muy bien! Intenta levantar un poco más tu pierna para lograr la posición perfecta'
            angle_rec.append(rec)
        else:
            ang_2_score=1
            rec='Inténtalo de nuevo, recuerda que debes levantar más la pierna ¡Tú puedes!'
            angle_rec.append(rec)
        
        if(mean_ang_1_o>min_thr_1):
            ang_1_score=1
            rec='Inténtalo de nuevo, debes doblar más la rodilla ¡Tú puedes!'
            angle_rec.append(rec)
        elif(mean_ang_1_o>mid_thr_1):
            ang_1_score=2
            rec='¡Muy bien! Intenta doblar un poco más tu rodilla '
            angle_rec.append(rec)
        else:
            ang_1_score=3
        
        


    elif (valid_time<time_thr_max):

        time_score=2
        rec='¡Muy bien! Lograste 15 segundos de mantención de elongación ¡Sigue así!'
        time_rec.append(rec)
        
        if(mean_ang_1>mid_thr_1):
            ang_1_score=2
            rec='¡Muy bien! Intenta doblar un poco más tu rodilla '
            angle_rec.append(rec)
        else:
            ang_1_score=3

        if(mean_ang_2>mid_thr_2):
            ang_2_score=2
            rec='¡Muy bien! Intenta levantar un poco más tu pierna para lograr la posición perfecta'
            angle_rec.append(rec)
        else:
            ang_2_score=3

    else:
        time_score=3
 
        if(mean_ang_1>mid_thr_1):
            ang_1_score=2
            rec='¡Muy bien! Intenta doblar un poco más tu rodilla '
            angle_rec.append(rec)
        else:
            ang_1_score=3

        if(mean_ang_2>mid_thr_2):
            ang_2_score=2
            rec='¡Muy bien! Intenta levantar un poco más tu pierna para lograr la posición perfecta'
            angle_rec.append(rec)
        else:
            ang_2_score=3

    if(ang_1_score<ang_2_score):
        ang_score=ang_1_score
    else:
        ang_score=ang_2_score

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


