# -*- coding: utf-8 -*-
import math
import numpy as np

def angulo_vertical(v1,v2):
    try:
        teta = math.degrees(math.atan(abs(v2[0]-v1[0])/abs(v2[1]-v1[1])))
    except:
        if v2[1] == v1[1]:
            teta = 90
    if v2[1]<v1[1]:
        teta = 180 - teta
    return teta

def angulo_horizontal(v1,v2):

    try:
        teta = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
    except:
        if v2[0] == v1[0]:
            teta = 90
    if v2[0]<v1[0]:
        teta = 180 - teta
    return teta

def ang_tresp(a,b,c):

    ang = math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]) - math.atan2(c[1]-a[1], c[0]-a[0]))
    return ang + 360 if ang < 0 else ang


def cuello_lateral(self):
  
    # Angulos de inclinacion del cuello

    ## Izquierda
    p1 = np.array(self.pose[0:2, 9])
    p2 = np.array(self.pose[0:2, 3])
    p3 = np.array(self.pose[0:2, 18])
    
    teta_i = ang_tresp(p1,p2,p3)

    ## Derecha
    p1 = np.array(self.pose[0:2, 3])
    p2 = np.array(self.pose[0:2, 17])
    p3 = np.array(self.pose[0:2, 9])
    
    teta_d = ang_tresp(p1,p2,p3)

    teta = teta_d - teta_i

    return [{'value':teta,'title':'Inclinacion'}]


def cuello_posterior(self):

    # Desde la izquierda

    np.seterr(divide='raise')

    v1 = tuple(self.pose[0:2, 1])
    v2 = tuple(self.pose[0:2, 3])
    try:
        teta = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
    except:
        if v2[0] == v1[0]:
            teta = 90
    return [{'value':teta,'title':'Flexion Cuello'}]

def brazo_izquierdo(self):
    
    # Angulos de hombro

    p1 = np.array(self.pose[0:2, 3])
    p2 = np.array(self.pose[0:2, 5])
    p3 = np.array(self.pose[0:2, 6])
    
    teta = ang_tresp(p1,p2,p3)

    return [{'value':teta,'title':'Brazo izq.'}]

def brazo_derecho(self):

    # Angulos de hombro

    p1 = np.array(self.pose[0:2, 9])
    p2 = np.array(self.pose[0:2, 12])
    p3 = np.array(self.pose[0:2, 11])
    
    teta = ang_tresp(p1,p2,p3)

    return [{'value':teta,'title':'Brazo izq.'}]

def circunduccion_hombro_izquierdo(self):
    
    # Angulos de hombro

    ## Izquierda
    p1 = np.array(self.pose[0:2, 3])
    p2 = np.array(self.pose[0:2, 18])
    p3 = np.array(self.pose[0:2, 4])
    
    teta = ang_tresp(p1,p2,p3)

    return [{'value':teta,'title':'Brazo izq.'}]

def circunduccion_hombro_derecho(self):
    
    # Angulos de hombro

    ## Izquierda
    p1 = np.array(self.pose[0:2, 9])
    p2 = np.array(self.pose[0:2, 17])
    p3 = np.array(self.pose[0:2, 11])
    
    teta = ang_tresp(p1,p2,p3)

    return [{'value':teta,'title':'Brazo der.'}]

def sep_brazos(self):

    ##Izquierdo
    p1 = np.array(self.pose[0:2, 3])
    p2 = np.array(self.pose[0:2, 6])
    p3 = np.array(self.pose[0:2, 5])
    
    
    teta_i = ang_tresp(p1,p2,p3)

    ## Derecho
    p1 = np.array(self.pose[0:2, 9])
    p2 = np.array(self.pose[0:2, 11])
    p3 = np.array(self.pose[0:2, 12])

    teta_d = ang_tresp(p1,p2,p3)

    return [{'value':teta_i,'title':'Codo Izq.'},{'value':teta_d,'title':'Codo Der.'}]

def circunduccion_codos(self):

    ##Izquierdo
    p1 = np.array(self.pose[0:2, 4])
    p2 = np.array(self.pose[0:2, 5])
    p3 = np.array(self.pose[0:2, 3])
    
    teta_i = ang_tresp(p1,p2,p3)

    ## Derecho
    p1 = np.array(self.pose[0:2, 10])
    p2 = np.array(self.pose[0:2, 9])
    p3 = np.array(self.pose[0:2, 11])

    teta_d = ang_tresp(p1,p2,p3)

    return [{'value':teta_i,'title':'Flexion Izq.'},{'value':teta_d,'title':'Flexion Der.'}]

def lateralizacion_columna(self):

    ##Izquierda
    #Brazo
    p1 = np.array(self.pose[0:2, 3])
    p2 = np.array(self.pose[0:2, 6])
    p3 = np.array(self.pose[0:2, 5])
    teta_ib = ang_tresp(p1,p2,p3)

    #Cadera
    p1 = np.array(self.pose[0:2, 6])
    p2 = np.array(self.pose[0:2, 8])
    p3 = np.array(self.pose[0:2, 3])
    teta_ic = ang_tresp(p1,p2,p3)

    ##Derecha
    #Brazo
    p1 = np.array(self.pose[0:2, 9])
    p2 = np.array(self.pose[0:2, 11])
    p3 = np.array(self.pose[0:2, 12])
    teta_db = ang_tresp(p1,p2,p3)

    #Cadera
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 9])
    p3 = np.array(self.pose[0:2, 14])
    teta_dc = ang_tresp(p1,p2,p3)

    return [{'value':teta_ib,'title':'Brazo izq.'},{'value':teta_ic,'title':'Cad izq.'},{'value':teta_db,'title':'Brazo der.'},{'value':teta_dc,'title':'Cad der.'}]

def rotacion_columna(self):
    #Pecho
    p1 = np.array(self.pose[0:2, 1])
    p2 = np.array(self.pose[0:2, 9])
    p3 = np.array(self.pose[0:2, 3])
    teta = ang_tresp(p1,p2,p3)

    if (p2[0]>p3[0]):
        teta = ang_tresp(p1,p3,p2)
    else:
        teta = ang_tresp(p1,p2,p3)

    if (teta>=200):
        teta=0
    return [{'value':teta,'title':'Rotacion.'}]


def circunduccion_caderas(self):

    # Angulos de inclinacion de la cadera

    ## Izquierda
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 14])
    p3 = np.array(self.pose[0:2, 8])
    
    teta_i = ang_tresp(p1,p2,p3)

    ## Derecha
    p1 = np.array(self.pose[0:2, 6])
    p2 = np.array(self.pose[0:2, 14])
    p3 = np.array(self.pose[0:2, 8])
    
    teta_d = ang_tresp(p1,p2,p3)

    teta = teta_d - teta_i

    return [{'value':teta,'title':'Inclinacion cadera'}]

def lateralizacion_caderas(self):

    # Angulos de inclinacion de la cadera

    ## Izquierda
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 14])
    p3 = np.array(self.pose[0:2, 8])
    
    teta_i = ang_tresp(p1,p2,p3)

    ## Derecha
    p1 = np.array(self.pose[0:2, 6])
    p2 = np.array(self.pose[0:2, 14])
    p3 = np.array(self.pose[0:2, 8])
    
    teta_d = ang_tresp(p1,p2,p3)

    teta = teta_d - teta_i

    return [{'value':teta,'title':'Inclinacion cadera'}]

def circunduccion_rodillas(self):

    # Angulos de inclinacion de la cadera

    ## Izquierda
    p1 = np.array(self.pose[0:2, 13])
    p2 = np.array(self.pose[0:2, 14])
    p3 = np.array(self.pose[0:2, 8])
    
    teta_i = ang_tresp(p1,p2,p3)

    ## Derecha
    p1 = np.array(self.pose[0:2, 7])
    p2 = np.array(self.pose[0:2, 14])
    p3 = np.array(self.pose[0:2, 8])
    
    teta_d = ang_tresp(p1,p2,p3)

    teta = teta_d - teta_i

    return [{'value':teta,'title':'Inclinacion rodillas'}]

def pecho(self):

    # Brazo
    p1 = np.array(self.pose[0:2, 9])
    p2 = np.array(self.pose[0:2, 11])
    p3 = np.array(self.pose[0:2, 12])
    
    teta_1 = ang_tresp(p1,p2,p3)

    # Columna
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 14])
    p3 = np.array(self.pose[0:2, 9])
    
    teta_2 = ang_tresp(p1,p2,p3)

    return [{'value':teta_1,'title':'Brazo'},{'value':teta_2,'title':'Columna'}]

def inclinacion_columna_izq(self):

    ##Derecha
    #Brazo
    p1 = np.array(self.pose[0:2, 9])
    p2 = np.array(self.pose[0:2, 11])
    p3 = np.array(self.pose[0:2, 12])
    teta_db = ang_tresp(p1,p2,p3)

    #Cadera
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 9])
    p3 = np.array(self.pose[0:2, 14])
    teta_dc = ang_tresp(p1,p2,p3)

    return [{'value':teta_db,'title':'Brazo der.'},{'value':teta_dc,'title':'Cad der.'}]


def inclinacion_columna_der(self):

    ##Izquierda
    #Brazo
    p1 = np.array(self.pose[0:2, 3])
    p2 = np.array(self.pose[0:2, 6])
    p3 = np.array(self.pose[0:2, 5])
    teta_ib = ang_tresp(p1,p2,p3)

    #Cadera
    p1 = np.array(self.pose[0:2, 6])
    p2 = np.array(self.pose[0:2, 8])
    p3 = np.array(self.pose[0:2, 3])
    teta_ic = ang_tresp(p1,p2,p3)

    return [{'value':teta_ib,'title':'Brazo izq.'},{'value':teta_ic,'title':'Cad izq.'}]

def flexion_columna(self):
    #Cadera izq
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 14])
    p3 = np.array(self.pose[0:2, 9])
    teta = ang_tresp(p1,p2,p3) 

    return [{'value':teta,'title':'Cadera.'}]

def gluteo_der(self):

    # Angulo rodilla
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 7])
    p3 = np.array(self.pose[0:2, 14])
    
    teta = ang_tresp(p1,p2,p3)

    return [{'value':teta,'title':'Rodilla D.'}]


def gluteo_izq(self):

     ##Izquierda
    p1 = np.array(self.pose[0:2, 6])
    p2 = np.array(self.pose[0:2, 8])
    p3 = np.array(self.pose[0:2, 13])
    
    teta = ang_tresp(p1,p2,p3)

    return [{'value':teta,'title':'Rodilla I.'}]

def cuadriceps_der(self):

    #Vista desde la Derecha

    ## Flexion cadera
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 13])
    p3 = np.array(self.pose[0:2, 9])

    teta_1 = ang_tresp(p1,p2,p3)

    ## Flexion rodilla
    p1 = np.array(self.pose[0:2, 13])
    p2 = np.array(self.pose[0:2, 12])
    p3 = np.array(self.pose[0:2, 14])

    teta_2 = ang_tresp(p1,p2,p3)

    return [{'value':teta_1,'title':'Flexion Cad.'},{'value':teta_2,'title':'Flexion Rod.'}]

def cuadriceps_izq(self):

    #Vista desde la Izquierda
    ## Flexion cadera
    p1 = np.array(self.pose[0:2, 6])
    p2 = np.array(self.pose[0:2, 7])
    p3 = np.array(self.pose[0:2, 3])
    teta_1 = ang_tresp(p1,p2,p3)

    ##Izquierda
    p1 = np.array(self.pose[0:2, 7])
    p2 = np.array(self.pose[0:2, 8])
    p3 = np.array(self.pose[0:2, 6])
    teta_2 = ang_tresp(p1,p2,p3)

    return [{'value':teta_1,'title':'Flexion Cad.'},{'value':teta_2,'title':'Flexion Rod.'}]

def isquiotibial_der(self):

    ##Derecha
    #Cadera
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 13])
    p3 = np.array(self.pose[0:2, 9])
    teta_c = ang_tresp(p1,p2,p3)

    #Rodilla
    p1 = np.array(self.pose[0:2, 13])
    p2 = np.array(self.pose[0:2, 12])
    p3 = np.array(self.pose[0:2, 14])
    teta_r = ang_tresp(p1,p2,p3) 

    return [{'value':teta_c,'title':'Cadera.'},{'value':teta_r,'title':'Rodilla'}]

def isquiotibial_izq(self):

    ##Izquierda
    #Cadera
    p1 = np.array(self.pose[0:2, 6])
    p2 = np.array(self.pose[0:2, 3])
    p3 = np.array(self.pose[0:2, 7])
    teta_c = ang_tresp(p1,p2,p3)

    #Rodilla
    p1 = np.array(self.pose[0:2, 7])
    p2 = np.array(self.pose[0:2, 8])
    p3 = np.array(self.pose[0:2, 6])
    teta_r = ang_tresp(p1,p2,p3) 

    return [{'value':teta_c,'title':'Cadera.'},{'value':teta_r,'title':'Rodilla'}]


def gemelos_izq(self):

    # Angulo cadera
    p1 = np.array(self.pose[0:2, 6])
    p2 = np.array(self.pose[0:2, 3])
    p3 = np.array(self.pose[0:2, 7])
    
    teta_1 = ang_tresp(p1,p2,p3)

    # Angulo rodilla
    p1 = np.array(self.pose[0:2, 7])
    p2 = np.array(self.pose[0:2, 6])
    p3 = np.array(self.pose[0:2, 8])
    
    teta_2 = ang_tresp(p1,p2,p3)

    # Angulo inclinacion
    p1 = np.array(self.pose[0:2, 3])
    p2 = np.array(self.pose[0:2, 8])
    p3 = np.array(self.pose[0:2, 14])
    
    teta_3 = ang_tresp(p1,p2,p3)

    return [{'value':teta_2,'title':'Rodilla I.'},{'value':teta_1,'title':'Cadera I.'},{'value':teta_3,'title':'Inclinacion'}]

def gemelos_der(self):

    # Angulo cadera
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 9])
    p3 = np.array(self.pose[0:2, 13])
    teta_1 = ang_tresp(p1,p2,p3)

    # Angulo rodilla
    p1 = np.array(self.pose[0:2, 13])
    p2 = np.array(self.pose[0:2, 12])
    p3 = np.array(self.pose[0:2, 14])
    teta_2 = ang_tresp(p1,p2,p3)

    # Angulo inclinacion
    p1 = np.array(self.pose[0:2, 9])
    p2 = np.array(self.pose[0:2, 14])
    p3 = np.array(self.pose[0:2, 8])
    
    teta_3 = ang_tresp(p1,p2,p3)

    return [{'value':teta_2,'title':'Rodilla D.'},{'value':teta_1,'title':'Cadera D.'},{'value':teta_3,'title':'Inclinacion'}]


def marcha_estacionaria(self):

    # Angulos de caderas

    ## Derecha
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 13])
    p3 = np.array(self.pose[0:2, 9])

    teta_d1 = ang_tresp(p1,p2,p3)


    # Angulos de rodilla

    ## Derecha
    p1 = np.array(self.pose[0:2, 13])
    p2 = np.array(self.pose[0:2, 12])
    p3 = np.array(self.pose[0:2, 14])

    teta_d2 = ang_tresp(p1,p2,p3)

    return [{'value':teta_d2,'title':'Rodilla D.'},{'value':teta_d1,'title':'Cadera D.'}]

def talon_gluteo(self):

    ## Derecha
    p1 = np.array(self.pose[0:2, 13])
    p2 = np.array(self.pose[0:2, 12])
    p3 = np.array(self.pose[0:2, 14])

    teta_d = ang_tresp(p1,p2,p3)

    return [{'value':teta_d,'title':'Rodilla D.'}]


def parar_sentar(self):

    # Angulos de cadera
    ## Izquierda
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 13])
    p3 = np.array(self.pose[0:2, 9])
    teta_1 = ang_tresp(p1,p2,p3)

    # Angulos de rodilla
    ##Izquierda
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 14])
    p3 = np.array(self.pose[0:2, 9])
    teta_2 = ang_tresp(p1,p2,p3)

    return [{'value':teta_2,'title':'Cadera'},{'value':teta_1,'title':'Columna'}]


def sentadilla_lat(self):
    
    # Angulos de rodilla

    ## Izquierda
    p1 = np.array(self.pose[0:2, 0])
    p2 = np.array(self.pose[0:2, 13])
    p3 = np.array(self.pose[0:2, 7])
    teta_1 = ang_tresp(p1,p2,p3)

    return [{'value':teta_1,'title':'Rodillas'}]

def rodilla_codo(self):

     ## Codo der - rodilla izq
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 7])
    p3 = np.array(self.pose[0:2, 10])
    teta_1 = ang_tresp(p1,p2,p3)

    ## Codo izq - rodilla der
    p1 = np.array(self.pose[0:2, 6])
    p2 = np.array(self.pose[0:2, 4])
    p3 = np.array(self.pose[0:2, 13])
    teta_2 = ang_tresp(p1,p2,p3)

    return [{'value':teta_1,'title':'CD-RI'},{'value':teta_2,'title':'CI-RD'}]

def brincos_jack(self):

    ## Levantamiento brazos
    p1 = np.array(self.pose[0:2, 3])
    p2 = np.array(self.pose[0:2, 11])
    p3 = np.array(self.pose[0:2, 12])
    teta_d = ang_tresp(p1,p2,p3)

    p1 = np.array(self.pose[0:2, 9])
    p2 = np.array(self.pose[0:2, 6])
    p3 = np.array(self.pose[0:2, 5])
    teta_i = ang_tresp(p1,p2,p3)

    teta_1 = teta_d+teta_i

    ## Separacion piernas
    p1 = np.array(self.pose[0:2, 0])
    p2 = np.array(self.pose[0:2, 14])
    p3 = np.array(self.pose[0:2, 8])
    teta_2 = ang_tresp(p1,p2,p3)

    return [{'value':teta_1,'title':'Brazos'},{'value':teta_2,'title':'Piernas'}]

def paso_adelante_atras(self):


    p1 = np.array(self.pose[0:2, 6])

    p3 = np.array(self.pose[0:2, 14])
    p2 = np.array(self.pose[0:2, 8])

    teta = ang_tresp(p1,p2,p3)

    return [{'value':teta,'title':'Desplazamiento*'}]

def boxing(self):

    ## Brazo
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 11])
    p3 = np.array(self.pose[0:2, 9])
    teta = ang_tresp(p1,p2,p3)

    return [{'value':teta,'title':'Hombro'}]

def sentadilla_unipodal_izquierda(self):

    ##rodilla izq
    p1 = np.array(self.pose[0:2, 7])
    p2 = np.array(self.pose[0:2, 8])
    p3 = np.array(self.pose[0:2, 6])
    teta_ri = ang_tresp(p1,p2,p3)

    ##Rodilla derecha
    p1 = np.array(self.pose[0:2, 13])
    p2 = np.array(self.pose[0:2, 14])
    p3 = np.array(self.pose[0:2, 12])
    teta_rd = ang_tresp(p1,p2,p3)
    
    ##Cadera izquierda
    p1 = np.array(self.pose[0:2, 6])
    p2 = np.array(self.pose[0:2, 7])
    p3 = np.array(self.pose[0:2, 3])
    teta_ci = ang_tresp(p1,p2,p3)

    ##Cadera derecha
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 9])
    p3 = np.array(self.pose[0:2, 13])
    teta_cd = ang_tresp(p1,p2,p3)

    return [{'value':teta_ri,'title':'Rod izq.'},{'value':teta_ci,'title':'Cad izq.'},{'value':teta_rd,'title':'Rod der'},{'value':teta_cd,'title':'Cad der.'}]

def sentadilla_unipodal_derecha(self):

    ##rodilla izq
    p1 = np.array(self.pose[0:2, 7])
    p2 = np.array(self.pose[0:2, 6])
    p3 = np.array(self.pose[0:2, 8])
    teta_ri = ang_tresp(p1,p2,p3)

    ##Rodilla derecha
    p1 = np.array(self.pose[0:2, 13])
    p2 = np.array(self.pose[0:2, 12])
    p3 = np.array(self.pose[0:2, 14])
    teta_rd = ang_tresp(p1,p2,p3)
    
    ##Cadera izquierda
    p1 = np.array(self.pose[0:2, 6])
    p2 = np.array(self.pose[0:2, 3])
    p3 = np.array(self.pose[0:2, 7])
    teta_ci = ang_tresp(p1,p2,p3)

    ##Cadera derecha
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 13])
    p3 = np.array(self.pose[0:2, 9])
    teta_cd = ang_tresp(p1,p2,p3)

    return [{'value':teta_rd,'title':'Rod der'},{'value':teta_cd,'title':'Cad der.'},{'value':teta_ri,'title':'Rod izq.'},{'value':teta_ci,'title':'Cad izq.'}]

def saltar_cuerda(self):

    ##rodilla der
    p1 = np.array(self.pose[0:2, 13])
    p2 = np.array(self.pose[0:2, 12])
    p3 = np.array(self.pose[0:2, 14])
    teta_ri = ang_tresp(p1,p2,p3)
    
    ##Codo der
    p1 = np.array(self.pose[0:2, 10])
    p2 = np.array(self.pose[0:2, 11])
    p3 = np.array(self.pose[0:2, 9])
    teta_cd = ang_tresp(p1,p2,p3)

    return [{'value':teta_ri,'title':'Rod der.'},{'value':teta_cd,'title':'Codo der.'}]

def rotacion_cuello(self):

    # Angulos de inclinacion del cuello

    ## Izquierda
    p1 = np.array(self.pose[0:2, 9])
    p2 = np.array(self.pose[0:2, 3])
    p3 = np.array(self.pose[0:2, 1])
    
    teta_i = ang_tresp(p1,p2,p3)

    ## Derecha
    p1 = np.array(self.pose[0:2, 3])
    p2 = np.array(self.pose[0:2, 1])
    p3 = np.array(self.pose[0:2, 9])
    
    teta_d = ang_tresp(p1,p2,p3)

    teta = teta_d - teta_i

    return [{'value':teta,'title':'Inclinacion'}]

def triceps(self):

    ## Codo der
    p1 = np.array(self.pose[0:2, 10])
    p2 = np.array(self.pose[0:2, 11])
    p3 = np.array(self.pose[0:2, 9])
    teta = ang_tresp(p1,p2,p3)

    return [{'value':teta,'title':'Codo.'}]

def aplauso(self):
    ## Derecho
    p1 = np.array(self.pose[0:2, 9])
    p2 = np.array(self.pose[0:2, 12])
    p3 = np.array(self.pose[0:2, 11]) 
    if(p3[0]>p1[0]):
        teta = -ang_tresp(p1,p2,p3)
    else:
        teta = ang_tresp(p1,p3,p2)

    return [{'value':teta,'title':'brazo'}]

def flexoextension_columna(self):

    ## cadera
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 13])
    p3 = np.array(self.pose[0:2, 9])
    teta = ang_tresp(p1,p2,p3)

    return [{'value':teta,'title':'Cadera'}]

def inclinacion_flexion(self):

    ## Apertura piernas
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 14])
    p3 = np.array(self.pose[0:2, 8])
    teta_1 = ang_tresp(p1,p2,p3)

    ## Apertura piernas
    p1 = np.array(self.pose[0:2, 6])
    p2 = np.array(self.pose[0:2, 14])
    p3 = np.array(self.pose[0:2, 8])
    teta_2 = ang_tresp(p1,p2,p3)

    teta=teta_1-teta_2

    return [{'value':teta,'title':'Desplazamiento'}]

def brazo_pierna(self):

    ## Apertura piernas
    p1 = np.array(self.pose[0:2, 9])
    p2 = np.array(self.pose[0:2, 14])
    p3 = np.array(self.pose[0:2, 6])
    teta_1 = ang_tresp(p1,p2,p3)

    p1 = np.array(self.pose[0:2, 3])
    p2 = np.array(self.pose[0:2, 12])
    p3 = np.array(self.pose[0:2, 8])
    teta_2 = ang_tresp(p1,p2,p3)

    teta_p=teta_1-teta_2

    ## Apertura brazos
    p1 = np.array(self.pose[0:2, 9])
    p2 = np.array(self.pose[0:2, 11])
    p3 = np.array(self.pose[0:2, 12])
    teta_1 = ang_tresp(p1,p2,p3)

    p1 = np.array(self.pose[0:2, 3])
    p2 = np.array(self.pose[0:2, 6])
    p3 = np.array(self.pose[0:2, 5])
    teta_2 = ang_tresp(p1,p2,p3)

    teta_b=teta_1-teta_2

    return [{'value':teta_b,'title':'Brazos'},{'value':teta_p,'title':'Piernas'}]

def marcha_adelante_atras(self):
    ## Cadera
    p1 = np.array(self.pose[0:2, 6])
    p2 = np.array(self.pose[0:2, 0])
    p3 = np.array(self.pose[0:2, 7])
    teta_c = ang_tresp(p1,p2,p3)

    ## Rodilla
    p1 = np.array(self.pose[0:2, 7])
    p2 = np.array(self.pose[0:2, 8])
    p3 = np.array(self.pose[0:2, 6])
    teta_r = ang_tresp(p1,p2,p3)

    return [{'value':teta_c,'title':'Cadera'},{'value':teta_r,'title':'Rodilla'}]

def mano_talon(self):
     ## mano der - pie izq
    p1 = np.array(self.pose[0:2, 6])
    p2 = np.array(self.pose[0:2, 11])
    p3 = np.array(self.pose[0:2, 8])
    teta_1 = ang_tresp(p1,p2,p3)

    ## mano izq - pie der
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 14])
    p3 = np.array(self.pose[0:2, 5])
    teta_2 = ang_tresp(p1,p2,p3)

    return [{'value':teta_1,'title':'MD-PI'},{'value':teta_2,'title':'MI-PD'}]

def flexbrazo_extcadera(self):

    ## Brazo
    p1 = np.array(self.pose[0:2, 9])
    p2 = np.array(self.pose[0:2, 12])
    p3 = np.array(self.pose[0:2, 11])
    teta_b = ang_tresp(p1,p2,p3)

    ## Cadera
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 14])
    p3 = np.array(self.pose[0:2, 9])
    teta_c = ang_tresp(p1,p2,p3)

    return [{'value':teta_b,'title':'Brazo'},{'value':teta_c,'title':'Cadera'}]

def extension_rodilla(self):

    ## Derecho
    p1 = np.array(self.pose[0:2, 13])
    p2 = np.array(self.pose[0:2, 12])
    p3 = np.array(self.pose[0:2, 14])
    teta_d = ang_tresp(p1,p2,p3)

    return [{'value':teta_d,'title':'Derecha'}]

def balance_pie_izq(self):
    
    # Movimiento del brazo izquierdo con respecto al pecho
    v1 = tuple(self.pose[0:2, 0])
    v2 = tuple(self.pose[0:2, 8])

    teta = angulo_vertical(v1,v2)
    if (v1[0]>v2[0]):
        teta = -teta

    return [{'value':teta,'title':'Pie'}]

def balance_pie_der(self):
    
    # Movimiento del brazo izquierdo con respecto al pecho
    v1 = tuple(self.pose[0:2, 0])
    v2 = tuple(self.pose[0:2, 14])

    teta = angulo_vertical(v1,v2)

    if (v1[0]>v2[0]):
        teta = -teta

    return [{'value':teta,'title':'Pie'}]


def sentadilla_frontal(self):

    # Angulos de rodillas

    p1 = np.array(self.pose[0:2, 0])
    p2 = np.array(self.pose[0:2, 13])
    p3 = np.array(self.pose[0:2, 7])
    teta_1 = ang_tresp(p1,p2,p3)

    return [{'value':teta_1,'title':'Rodillas'}]

def sentadilla(self):

    # Angulos de cadera
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 13])
    p3 = np.array(self.pose[0:2, 9])
    teta_1 = ang_tresp(p1,p2,p3)

    # Angulos de rodilla
    p1 = np.array(self.pose[0:2, 13])
    p2 = np.array(self.pose[0:2, 12])
    p3 = np.array(self.pose[0:2, 14])
    teta_2 = ang_tresp(p1,p2,p3)

    return [{'value':teta_2,'title':'Rodilla'},{'value':teta_1,'title':'Cadera'}]

def apoyo_unipodal_der(self):
    # Angulos de cadera
    p1 = np.array(self.pose[0:2, 6])
    p2 = np.array(self.pose[0:2, 3])
    p3 = np.array(self.pose[0:2, 7])
    teta_1 = ang_tresp(p1,p2,p3)

    # Angulos de rodilla
    p1 = np.array(self.pose[0:2, 7])
    p2 = np.array(self.pose[0:2, 8])
    p3 = np.array(self.pose[0:2, 6])
    teta_2 = ang_tresp(p1,p2,p3)

    return [{'value':teta_2,'title':'Rodilla'},{'value':teta_1,'title':'Cadera'}]


def apoyo_unipodal_izq(self):
    
    # Angulos de cadera
    p1 = np.array(self.pose[0:2, 12])
    p2 = np.array(self.pose[0:2, 13])
    p3 = np.array(self.pose[0:2, 9])
    teta_1 = ang_tresp(p1,p2,p3)

    # Angulos de rodilla
    p1 = np.array(self.pose[0:2, 13])
    p2 = np.array(self.pose[0:2, 12])
    p3 = np.array(self.pose[0:2, 14])
    teta_2 = ang_tresp(p1,p2,p3)

    return [{'value':teta_2,'title':'Rodilla'},{'value':teta_1,'title':'Cadera'}]

def flex_codo(self):
    # Angulo de codo
    p1 = np.array(self.pose[0:2, 10])
    p2 = np.array(self.pose[0:2, 11])
    p3 = np.array(self.pose[0:2, 9])
    teta = ang_tresp(p1,p2,p3)

    return [{'value':teta,'title':'Codo'}]

def codo_90(self):
    # Angulos de cadera
    p1 = np.array(self.pose[0:2, 9])
    p2 = np.array(self.pose[0:2, 11])
    p3 = np.array(self.pose[0:2, 10])

    if(p2[1]>=p3[1]):
        teta_1 = ang_tresp(p1,p3,p2)
    else:
        teta_1 = -ang_tresp(p1,p2,p3)

    # Angulos de rodilla
    p1 = np.array(self.pose[0:2, 3])
    p2 = np.array(self.pose[0:2, 5])
    p3 = np.array(self.pose[0:2, 4])

    if(p2[1]<=p3[1]):
        teta_2 = -ang_tresp(p1,p3,p2)
    else:
        teta_2 = ang_tresp(p1,p2,p3)
    
    if(teta_1>100 or teta_1<-100):
        teta_1=0

    if(teta_2>100 or teta_2<-100):
        teta_2=0

    return [{'value':teta_2,'title':'izq'},{'value':teta_1,'title':'der'}]

def abduccion_cadera(self):

    # Angulo de codo
    p1 = np.array(self.pose[0:2, 6])
    p2 = np.array(self.pose[0:2, 8])
    p3 = np.array(self.pose[0:2, 3])
    teta = ang_tresp(p1,p2,p3)

    return [{'value':teta,'title':'cadera'}]



class Exercises:
    angle_calculation = {
                        'M_CL'  : cuello_lateral,
                        'M_CP'  : cuello_posterior,
                        'M_CHI'  : circunduccion_hombro_izquierdo,
                        'M_CHD'  : circunduccion_hombro_derecho,
                        'M_SB'  : sep_brazos,
                        'M_CCO' : circunduccion_codos,
                        'M_RC'  : rotacion_columna,
                        'M_LC'  : lateralizacion_columna,
                        'M_CCA' : circunduccion_caderas,
                        'M_LCA' : lateralizacion_caderas,
                        'M_CR'  : circunduccion_rodillas,

                        'E_CLI' : cuello_lateral,
                        'E_CLD' : cuello_lateral,
                        'E_CP'  : cuello_posterior,
                        'E_BI'  : brazo_izquierdo,
                        'E_BD'  : brazo_derecho,
                        'E_P'   : pecho,
                        'E_ICD' : inclinacion_columna_der,
                        'E_ICI' : inclinacion_columna_izq,
                        'E_FC'  : flexion_columna,
                        'E_GD'  : gluteo_der,
                        'E_GI'  : gluteo_izq,
                        'E_CD'  : cuadriceps_der,
                        'E_CI'  : cuadriceps_izq,
                        'E_ID'  : isquiotibial_der,
                        'E_II'  : isquiotibial_izq,
                        'E_GED' : gemelos_der,
                        'E_GEI' : gemelos_izq,

                        'D_ME'  : marcha_estacionaria,
                        'D_TG'  : talon_gluteo,
                        'D_SL'  : sentadilla_lat,
                        'D_RC'  : rodilla_codo,
                        'D_BJ'  : brincos_jack,
                        'D_AA'  : paso_adelante_atras,
                        'D_B'   : boxing,
                        'D_SUD' : sentadilla_unipodal_derecha,
                        'D_SUI' : sentadilla_unipodal_izquierda,
                        'D_SC'  : saltar_cuerda,

                        'A_RC'  : rotacion_cuello,
                        'A_T'   : triceps,
                        'A_A'   : aplauso,
                        'A_FC'  : flexoextension_columna,
                        'A_IF'  : inclinacion_flexion,
                        'A_BP'  : brazo_pierna,
                        'A_AA'  : marcha_adelante_atras,
                        'A_MT'  : mano_talon,
                        'A_FE'  : flexbrazo_extcadera,
                        'A_ER'  : extension_rodilla,
                        'A_BPI' : balance_pie_izq,
                        'A_BPD' : balance_pie_der,
                        'A_PS'  : parar_sentar,
                        
                        'F_SF'  : sentadilla_frontal,
                        'F_SL'  : sentadilla,
                        'F_AUI' : apoyo_unipodal_izq,
                        'F_AUD' : apoyo_unipodal_der,

                        'L_FCODO': flex_codo,
                        'L_CODO90': codo_90,
                        'L_ABDCAD': abduccion_cadera
                        }

    def __init__(self, tag, pose):
        self.tag = tag
        self.pose = pose

    def calculate(self):
        if self.tag not in self.angle_calculation:
            return None
        return self.angle_calculation[self.tag](self)
