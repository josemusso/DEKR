import pandas as pd
import numpy as np
import math


def score(data):
    ## Calification rules

    #Arm
    max_border_1 = 180     # Max threshold for valid point
    min_border_1 = 0      # Min threshold for valid point

    min_thr_1 = 15
    mid_thr_1 = 25

    #Hip
    max_border_2 = 270     # Max threshold for valid point
    min_border_2 = 90      # Min threshold for valid point
    base_ang_2 = 180        # Objective angle
    ang_width_2 = 20     # Valid angles thresholds (%)

    time_thr_min=20       # Minimun
    time_thr_max=25       # MAx

    # Read data
    df = data

    #Order data
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
    df_1 = df[(df.ang_1>=min_thr_1)] #Select by 1st angle

    df_2 = df_1[(df_1.ang_2<=base_ang_2+ang_width_2) & (df_1.ang_2>=base_ang_2-ang_width_2)] #Select by 2nd angle

    #Calculate valid exercise time
    valid_time = round(df_2.frame_t.sum(),2)

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

    """ mean_ang_1=20
    valid_time=22

    print(mean_ang_1_o)
    print(mean_ang_2_o)
    print(valid_time) """

    if(valid_time<time_thr_min):
        score = False
        time_score=1
        rec='Inténtalo de nuevo, recuerda mantener la posición el máximo tiempo posible.'
        time_rec.append(rec)

        if((mean_ang_2_o<=base_ang_2+ang_width_2) and (mean_ang_2_o>=base_ang_2-ang_width_2)):
            ang_2_score=3
        else:
            ang_2_score=1
            rec='Inténtalo de nuevo, es importante mantener el tronco recto y erguido ¡Tú puedes!'
            angle_rec.append(rec)
        
        if(mean_ang_1_o<min_thr_1):
            ang_1_score=1
            rec='¡Falta poco! Busca subir más los brazos para alcanzar la posición ideal ¡Tú puedes!'
            angle_rec.append(rec)
        elif(mean_ang_1_o<mid_thr_1):
            ang_1_score=2
            rec='¡Vas por muy buen camino! Intenta levantar más los brazos'
            angle_rec.append(rec)
        else:
            ang_1_score=3

    elif (valid_time<time_thr_max):

        time_score=2
        rec='¡Muy bien! Lograste 20 segundos de mantención de elongación ¡Sigue así!'
        time_rec.append(rec)

        ang_2_score=3
        
        if(mean_ang_1<mid_thr_1):
            ang_1_score=2
            rec='¡Vas por muy buen camino! Intenta levantar más los brazos'
            angle_rec.append(rec)
        else:
            ang_1_score=3
    else:
        time_score=3
        ang_2_score=3

        if(mean_ang_1<mid_thr_1):
            ang_1_score=2
            rec='¡Vas por muy buen camino! Intenta levantar más los brazos'
            angle_rec.append(rec)
        else:
            ang_1_score=3

    #Angle score
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


