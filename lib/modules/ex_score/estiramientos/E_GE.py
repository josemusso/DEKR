import pandas as pd
import numpy as np
import math


def score(data):
    ## Calification rules

    #Knee
    max_border_3 = 270     # Max threshold for valid point
    min_border_3 = 90      # Min threshold for valid point
    max_ang_3=220
    min_ang_3=140

    #Hip
    max_border_2 = 270     # Max threshold for valid point
    min_border_2 = 90      # Min threshold for valid point
    max_ang_2=220   
    min_ang_2=140

    #Inclination
    max_border_1 = 90     # Max threshold for valid point
    min_border_1 = 0      # Min threshold for valid point

    max_thr_1=65
    mid_thr_1=45

    time_thr_min=20
    time_thr_max=25      # Minimun valid time

    # Read data
    df = data

    #Orde data
    df=df[['Second','Angle']]
    df['time']=df['Second']

    df[['ang_1','ang_2','ang_3']]=pd.DataFrame(df["Angle"].to_list(), columns=['ang_1','ang_2','ang_3'])
    df['ang_1']=df['ang_1'].astype(float)
    df['ang_2']=df['ang_2'].astype(float)
    df['ang_3']=df['ang_3'].astype(float)

    df.drop(columns=['Angle', 'Second'], inplace=True)


    #Fix bad frames
    df.loc[df['ang_1'] > max_border_1, 'ang_1'] = df.ang_1.mean()
    df.loc[df['ang_1'] < min_border_1, 'ang_1'] = df.ang_1.mean()

    df.loc[df['ang_2'] > max_border_2, 'ang_2'] = df.ang_2.mean()
    df.loc[df['ang_2'] < min_border_2, 'ang_2'] = df.ang_2.mean()

    df.loc[df['ang_3'] > max_border_3, 'ang_3'] = df.ang_3.mean()
    df.loc[df['ang_3'] < min_border_3, 'ang_3'] = df.ang_3.mean()

    #Calculate each frame rate
    df['frame_t'] = abs(df.time - df.time.shift(1))
    fps=round(1/df.frame_t.median(),2)

    #Save original stats
    df_o_stats=df.describe().astype(float).round(3).to_dict()

    #Get valid angles
    df_1 = df[(df.ang_1<=max_thr_1)] #Select by 1st angle

    df_2 = df_1[(df_1.ang_2<=max_ang_2) & (df_1.ang_2>=min_ang_2)] #Select by 2nd angle

    df_3 = df_2[(df_2.ang_3<=max_ang_3) & (df_2.ang_3>=min_ang_3)] #Select by 3rd angle

    #Calculate valid exercise time
    valid_time = round(df_3.frame_t.sum(),2)

    #Save final stats
    df_stats=df_3.describe().astype(float).round(3).to_dict()

    #Evaluate
    score = True
    time_rec=[]
    angle_rec=[]

    mean_ang_1_o=df.ang_1.mean()
    mean_ang_2_o=df.ang_2.mean()
    mean_ang_3_o=df.ang_3.mean()

    mean_ang_1=df_3.ang_1.mean()
    #mean_ang_2=df_3.ang_2.mean()
    #mean_ang_3=df_3.ang_3.mean()

    """ mean_ang_1_o=65
    mean_ang_2_o=230
    mean_ang_3_o=120
    valid_time=21

    print(mean_ang_1_o)
    print(mean_ang_2_o)
    print(valid_time) """


    if(valid_time<time_thr_min):
        score = False
        time_score=1
        rec='Inténtalo de nuevo, recuerda mantener la posición el máximo tiempo posible.'
        time_rec.append(rec)

        if((mean_ang_2_o<=max_ang_2) and (mean_ang_2_o>=min_ang_2)):
            ang_2_score=3
        else:
            ang_2_score=1
            rec='Inténtalo de nuevo, es importante mantener el tronco recto y erguido ¡Tú puedes!'
            angle_rec.append(rec)
        
        if((mean_ang_3_o<=max_ang_3) and (mean_ang_3_o>=min_ang_3)):
            ang_3_score=3
        else:
            ang_3_score=1
            rec='Inténtalo de nuevo, es importante que la rodilla de la pierna de atrás se mantenga estirada ¡Tú puedes!'
            angle_rec.append(rec)
        
        if(mean_ang_1_o>max_thr_1):
            ang_1_score=1
            rec='Inténtalo de nuevo, es importante llevar la pelvis hacia adelante manteniendo el troco erguido ¡Tú puedes!'
            angle_rec.append(rec)
        elif(mean_ang_1_o>mid_thr_1):
            ang_1_score=2
            rec='¡Muy bien! Intenta llevar la pelvis un poco más hacia adelante'
            angle_rec.append(rec)
        else:
            ang_1_score=3
        


    elif (valid_time<time_thr_max):

        time_score=2
        ang_2_score=3
        ang_3_score=3
        rec='¡Muy bien! Lograste 20 segundos de mantención de elongación ¡Sigue así!'
        time_rec.append(rec)
        
        if(mean_ang_1>max_thr_1):
            ang_1_score=1
            rec='Inténtalo de nuevo, es importante llevar la pelvis hacia adelante manteniendo el troco erguido ¡Tú puedes!'
            angle_rec.append(rec)
        elif(mean_ang_1>mid_thr_1):
            ang_1_score=2
            rec='¡Muy bien! Intenta llevar la pelvis un poco más hacia adelante'
            angle_rec.append(rec)
        else:
            ang_1_score=3
    else:
        time_score=3

        ang_2_score=3
        ang_3_score=3

        if(mean_ang_1>max_thr_1):
            ang_1_score=1
            rec='Inténtalo de nuevo, es importante llevar la pelvis hacia adelante manteniendo el troco erguido ¡Tú puedes!'
            angle_rec.append(rec)
        elif(mean_ang_1>mid_thr_1):
            ang_1_score=2
            rec='¡Muy bien! Intenta llevar la pelvis un poco más hacia adelante'
            angle_rec.append(rec)
        else:
            ang_1_score=3

    if(ang_1_score==1 or ang_2_score==1 or ang_3_score==1):
        ang_score=1
    elif(ang_1_score==2 or ang_2_score==2 or ang_3_score==2):
        ang_score=2
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

