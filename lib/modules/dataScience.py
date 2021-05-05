# -*- coding: utf-8 -*-
import math
import numpy as np


def cadera_ext(self):
    v1 = tuple(self.pose[0:2, 6].astype(np.int32))
    v2 = tuple(self.pose[0:2, 7].astype(np.int32))
    try:
        teta = math.degrees(math.atan(abs(v2[0]-v1[0])/abs(v2[1]-v1[1])))
    except:
        if v2[1] == v1[1]:
            teta = 90
    # Angulo rodilla
    p1 = tuple(self.pose[0:2, 7].astype(np.int32))
    p2 = tuple(self.pose[0:2, 8].astype(np.int32))
    p3 = tuple(self.pose[0:2, 6].astype(np.int32))
    try:
        v1 = (p1[0], -p1[1])
        v2 = (p2[0], -p2[1])
        v3 = (p3[0], -p3[1])
        a = np.array([[v2[0]-v1[0]], [v2[1]-v1[1]], [0]])
        b = np.array([[v3[0]-v1[0]], [v3[1]-v1[1]], [0]])
        c = np.cross(a, b, axis=0)
        m1 = (v2[1]-v1[1])/(v2[0]-v1[0])
        m2 = (v3[1]-v1[1])/(v3[0]-v1[0])
        teta2 = math.degrees(math.atan(((m2-m1)/(1+m2*m1))))
        if teta2 > 0 and c[2][0] > 0:
            pass
        elif teta2 > 0 and c[2][0] < 0:
            teta2 += 180
        elif teta2 < 0 and c[2][0] > 0:
            teta2 += 180
        elif teta2 < 0 and c[2][0] < 0:
            teta2 += 360
    except:
        if v1[0] == v2[0] == v3[0]:
            teta2 = 180
        elif v1[0] == v2[0] != v3[0]:
            # calculate
            pass
        elif v1[0] == v2[0] == v3[0]:
            # calculate
            pass
        elif 1+m2*m1 == 0:
            teta2 = 90
    return [{'value': teta, 'title': 'Extension'}, {'value': teta2, 'title': 'Rodilla'}]


def cadera_flex(self):
    v1 = tuple(self.pose[0:2, 12].astype(np.int32))
    v2 = tuple(self.pose[0:2, 13].astype(np.int32))
    try:
        teta = math.degrees(math.atan(abs(v2[0]-v1[0])/abs(v2[1]-v1[1])))
    except:
        if v2[1] == v1[1]:
            teta = 90
    # Angulo rodilla
    p1 = tuple(self.pose[0:2, 13].astype(np.int32))
    p2 = tuple(self.pose[0:2, 14].astype(np.int32))
    p3 = tuple(self.pose[0:2, 12].astype(np.int32))
    try:
        v1 = (p1[0], -p1[1])
        v2 = (p2[0], -p2[1])
        v3 = (p3[0], -p3[1])
        a = np.array([[v2[0]-v1[0]], [v2[1]-v1[1]], [0]])
        b = np.array([[v3[0]-v1[0]], [v3[1]-v1[1]], [0]])
        c = np.cross(a, b, axis=0)
        m1 = (v2[1]-v1[1])/(v2[0]-v1[0])
        m2 = (v3[1]-v1[1])/(v3[0]-v1[0])
        teta2 = math.degrees(math.atan(((m2-m1)/(1+m2*m1))))
        if teta2 > 0 and c[2][0] > 0:
            pass
        elif teta2 > 0 and c[2][0] < 0:
            teta2 += 180
        elif teta2 < 0 and c[2][0] > 0:
            teta2 += 180
        elif teta2 < 0 and c[2][0] < 0:
            teta2 += 360
    except:
        if v1[0] == v2[0] == v3[0]:
            teta2 = 180
        elif v1[0] == v2[0] != v3[0]:
            # calculate
            pass
        elif v1[0] == v2[0] == v3[0]:
            # calculate
            pass
        elif 1+m2*m1 == 0:
            teta2 = 90
    teta2 = 360 - teta2
    return [{'value': teta, 'title': 'Extension'}, {'value': teta2, 'title': 'Rodilla'}]


def cadera_rot(self):
    v1 = tuple(self.pose[0:2, 13].astype(np.int32))
    v2 = tuple(self.pose[0:2, 14].astype(np.int32))
    try:
        teta = math.degrees(math.atan(abs(v2[0]-v1[0])/abs(v2[1]-v1[1])))
    except:
        if v2[1] == v1[1]:
            teta = 90
    return [{'value': teta, 'title': 'Interno'}, {'value': -teta, 'title': 'Externo'}]


def rodilla_flex(self):
    v1 = tuple(self.pose[0:2, 6].astype(np.int32))
    v2 = tuple(self.pose[0:2, 7].astype(np.int32))
    try:
        teta = math.degrees(math.atan(abs(v2[0]-v1[0])/abs(v2[1]-v1[1])))
    except:
        if v2[1] == v1[1]:
            teta = 90
    # Angulo rodilla
    p1 = tuple(self.pose[0:2, 7].astype(np.int32))
    p2 = tuple(self.pose[0:2, 8].astype(np.int32))
    p3 = tuple(self.pose[0:2, 6].astype(np.int32))
    try:
        v1 = (p1[0], -p1[1])
        v2 = (p2[0], -p2[1])
        v3 = (p3[0], -p3[1])
        a = np.array([[v2[0]-v1[0]], [v2[1]-v1[1]], [0]])
        b = np.array([[v3[0]-v1[0]], [v3[1]-v1[1]], [0]])
        c = np.cross(a, b, axis=0)
        m1 = (v2[1]-v1[1])/(v2[0]-v1[0])
        m2 = (v3[1]-v1[1])/(v3[0]-v1[0])
        teta2 = math.degrees(math.atan(((m2-m1)/(1+m2*m1))))
        if teta2 > 0 and c[2][0] > 0:
            pass
        elif teta2 > 0 and c[2][0] < 0:
            teta2 += 180
        elif teta2 < 0 and c[2][0] > 0:
            teta2 += 180
        elif teta2 < 0 and c[2][0] < 0:
            teta2 += 360
    except:
        if v1[0] == v2[0] == v3[0]:
            teta2 = 180
        elif v1[0] == v2[0] != v3[0]:
            # calculate
            pass
        elif v1[0] == v2[0] == v3[0]:
            # calculate
            pass
        elif 1+m2*m1 == 0:
            teta2 = 90
    return [{'value': teta2, 'title': 'Rodilla'}, {'value': teta, 'title': 'Cadera'}]


def craneo_cerv_flex(self):
    v1 = tuple(self.pose[0:2, 1].astype(np.int32))
    v2 = tuple(self.pose[0:2, 18].astype(np.int32))
    try:
        teta = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
    except:
        if v2[0] == v1[0]:
            teta = 90
    return [{'value': teta, 'title': 'Flexion'}]


def craneo_cerv_incl(self):
    v1 = tuple(self.pose[0:2, 18].astype(np.int32))
    v2 = tuple(self.pose[0:2, 17].astype(np.int32))
    try:
        teta = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
    except:
        if v2[0] == v1[0]:
            teta = 90
    return [{'value': teta, 'title': 'Inclinacion'}]


def hombro_flex(self):
    v1 = tuple(self.pose[0:2, 3].astype(np.int32))
    v2 = tuple(self.pose[0:2, 4].astype(np.int32))
    try:
        teta = math.degrees(math.atan(abs(v2[0]-v1[0])/abs(v2[1]-v1[1])))
    except:
        if v2[1] == v1[1]:
            teta = 90
    if v2[1] < v1[1]:
        teta = 180 - teta
    return [{'value': teta, 'title': 'Flexion'}]


def hombro_abdu(self):
    v1 = tuple(self.pose[0:2, 9].astype(np.int32))
    v2 = tuple(self.pose[0:2, 10].astype(np.int32))
    try:
        teta = math.degrees(math.atan(abs(v2[0]-v1[0])/abs(v2[1]-v1[1])))
    except:
        if v2[1] == v1[1]:
            teta = 90
    if v2[1] < v1[1]:
        teta = 180 - teta
    return [{'value': teta, 'title': 'Abduccion'}]


def hombro_ext(self):
    v1 = tuple(self.pose[0:2, 3].astype(np.int32))
    v2 = tuple(self.pose[0:2, 4].astype(np.int32))
    try:
        teta = math.degrees(math.atan(abs(v2[0]-v1[0])/abs(v2[1]-v1[1])))
    except:
        if v2[1] == v1[1]:
            teta = 90
    if v2[1] < v1[1]:
        teta = 180 - teta
    return [{'value': teta, 'title': 'Extension'}]


def hombro_rot_int(self):
    v1 = tuple(self.pose[0:2, 4].astype(np.int32))
    v2 = tuple(self.pose[0:2, 5].astype(np.int32))
    try:
        teta = math.degrees(math.atan(abs(v2[0]-v1[0])/abs(v2[1]-v1[1])))
    except:
        if v2[1] == v1[1]:
            teta = 90
    if v2[1] > v1[1]:
        teta = 180 - teta
    return [{'value': teta, 'title': 'Rotacion'}]


def hombro_rot_ext(self):
    v1 = tuple(self.pose[0:2, 4].astype(np.int32))
    v2 = tuple(self.pose[0:2, 5].astype(np.int32))
    try:
        teta = math.degrees(math.atan(abs(v2[0]-v1[0])/abs(v2[1]-v1[1])))
    except:
        if v2[1] == v1[1]:
            teta = 90
    if v2[1] < v1[1]:
        teta = 180 - teta
    return [{'value': teta, 'title': 'Rotacion'}]


def codo_flex(self):
    p1 = tuple(self.pose[0:2, 4].astype(np.int32))
    p2 = tuple(self.pose[0:2, 5].astype(np.int32))
    p3 = tuple(self.pose[0:2, 3].astype(np.int32))
    try:
        v1 = (p1[0], -p1[1])
        v2 = (p2[0], -p2[1])
        v3 = (p3[0], -p3[1])
        a = np.array([[v2[0]-v1[0]], [v2[1]-v1[1]], [0]])
        b = np.array([[v3[0]-v1[0]], [v3[1]-v1[1]], [0]])
        c = np.cross(a, b, axis=0)
        m1 = (v2[1]-v1[1])/(v2[0]-v1[0])
        m2 = (v3[1]-v1[1])/(v3[0]-v1[0])
        teta2 = math.degrees(math.atan(((m2-m1)/(1+m2*m1))))
        if teta2 > 0 and c[2][0] > 0:
            pass
        elif teta2 > 0 and c[2][0] < 0:
            teta2 += 180
        elif teta2 < 0 and c[2][0] > 0:
            teta2 += 180
        elif teta2 < 0 and c[2][0] < 0:
            teta2 += 360
    except:
        if v1[0] == v2[0] == v3[0]:
            teta2 = 180
        elif v1[0] == v2[0] != v3[0]:
            # calculate
            pass
        elif v1[0] == v2[0] == v3[0]:
            # calculate
            pass
        elif 1+m2*m1 == 0:
            teta2 = 90
    teta2 = 360 - teta2
    return [{'value': teta2, 'title': 'Flexion'}]


def angulo_vertical(v1, v2):
    try:
        teta = math.degrees(math.atan(abs(v2[0]-v1[0])/abs(v2[1]-v1[1])))
    except:
        if v2[1] == v1[1]:
            teta = 90
    if v2[1] < v1[1]:
        teta = 180 - teta
    return teta


def ang_tresp(a, b, c):

    ang = math.degrees(math.atan2(
        b[1]-a[1], b[0]-a[0]) - math.atan2(c[1]-a[1], c[0]-a[0]))
    return ang + 360 if ang < 0 else ang


def sep_hombros(self):

    np.seterr(divide='raise')

    # Codo
    # Derecho
    v1 = tuple(self.pose[0:2, 9].astype(np.int32))
    v2 = tuple(self.pose[0:2, 10].astype(np.int32))

    teta_1d = angulo_vertical(v1, v2)

    # Izquierdo
    v1 = tuple(self.pose[0:2, 3].astype(np.int32))
    v2 = tuple(self.pose[0:2, 4].astype(np.int32))

    teta_1i = angulo_vertical(v1, v2)

    # Muñeca
    # Derecho
    v1 = tuple(self.pose[0:2, 9].astype(np.int32))
    v2 = tuple(self.pose[0:2, 11].astype(np.int32))

    teta_2d = angulo_vertical(v1, v2)

    # Izquierdo
    v1 = tuple(self.pose[0:2, 3].astype(np.int32))
    v2 = tuple(self.pose[0:2, 5].astype(np.int32))

    teta_2i = angulo_vertical(v1, v2)

    return [{'value': teta_1d, 'title': 'Codo Der.'}, {'value': teta_2d, 'title': 'Mano Der.'}, {'value': teta_1i, 'title': 'Codo Izq.'}, {'value': teta_2i, 'title': 'Mano Izq.'}]


def elev_rodillas(self):

    np.seterr(divide='raise')

    # Angulos de cadera

    # Izquierda
    v1 = tuple(self.pose[0:2, 6].astype(np.int32))
    v2 = tuple(self.pose[0:2, 7].astype(np.int32))
    try:

        # Angulo de flexion de cadera
        teta_i1 = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))

        if v2[1] >= v1[1]:
            teta_i1 += 90  # Rodilla bajo de la cadera
        else:
            teta_i1 = 90-teta_i1  # Rodilla por encima de la cadera

    except:
        if v2[0] == v1[0]:
            teta_i1 = 180

    # Derecha
    v1 = tuple(self.pose[0:2, 12].astype(np.int32))
    v2 = tuple(self.pose[0:2, 13].astype(np.int32))
    try:
        teta_d1 = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))

        if v2[1] >= v1[1]:
            teta_d1 += 90  # Rodilla bajo de la cadera
        else:
            teta_d1 = 90-teta_d1  # Rodilla por encima de la cadera

    except:
        if v2[0] == v1[0]:
            teta_d1 = 180

    # Angulos de rodilla

    # Izquierda
    p1 = np.array(self.pose[0:2, 7].astype(np.int32))
    p2 = np.array(self.pose[0:2, 8].astype(np.int32))
    p3 = np.array(self.pose[0:2, 6].astype(np.int32))

    teta_i2 = ang_tresp(p1, p2, p3)

    # Derecha
    p1 = np.array(self.pose[0:2, 13].astype(np.int32))
    p2 = np.array(self.pose[0:2, 14].astype(np.int32))
    p3 = np.array(self.pose[0:2, 12].astype(np.int32))

    teta_d2 = ang_tresp(p1, p2, p3)

    return [{'value': teta_d2, 'title': 'Rodilla D.'}, {'value': teta_d1, 'title': 'Cadera D.'}, {'value': teta_i2, 'title': 'Rodilla I.'}, {'value': teta_i1, 'title': 'Cadera I.'}]


def talon_gluteo(self):

    np.seterr(divide='raise')

    # Angulos de rodilla

    # Izquierda
    p1 = np.array(self.pose[0:2, 7].astype(np.int32))
    p2 = np.array(self.pose[0:2, 8].astype(np.int32))
    p3 = np.array(self.pose[0:2, 6].astype(np.int32))

    teta_i = ang_tresp(p1, p2, p3)

    # Derecha
    p1 = np.array(self.pose[0:2, 13].astype(np.int32))
    p2 = np.array(self.pose[0:2, 14].astype(np.int32))
    p3 = np.array(self.pose[0:2, 12].astype(np.int32))

    teta_d = ang_tresp(p1, p2, p3)

    return [{'value': teta_d, 'title': 'Rodilla D.'}, {'value': teta_i, 'title': 'Rodilla I.'}]


def parar_sentar(self):

    np.seterr(divide='raise')
    # Vista desde la Izquierda
    # Pecho - cadera
    v1 = tuple(self.pose[0:2, 0].astype(np.int32))
    v2 = tuple(self.pose[0:2, 6].astype(np.int32))

    teta_1 = angulo_vertical(v1, v2)

    #Cadera - rodilla
    v1 = tuple(self.pose[0:2, 6].astype(np.int32))
    v2 = tuple(self.pose[0:2, 7].astype(np.int32))

    teta_2 = angulo_vertical(v1, v2)

    #Rodilla - talon
    v1 = tuple(self.pose[0:2, 7].astype(np.int32))
    v2 = tuple(self.pose[0:2, 8].astype(np.int32))

    teta_3 = angulo_vertical(v1, v2)

    return [{'value': teta_1, 'title': 'Pecho-Cad'}, {'value': teta_2, 'title': 'Cad-Rod'}, {'value': teta_3, 'title': 'Rod-Talon'}]


def sentadilla(self):

    np.seterr(divide='raise')

    # Vista desde la Izquierda
    # Hombro - cadera
    v1 = tuple(self.pose[0:2, 0].astype(np.int32))
    v2 = tuple(self.pose[0:2, 6].astype(np.int32))

    teta_1 = angulo_vertical(v1, v2)

    #Cadera - rodilla
    v1 = tuple(self.pose[0:2, 6].astype(np.int32))
    v2 = tuple(self.pose[0:2, 7].astype(np.int32))

    teta_2 = angulo_vertical(v1, v2)

    #Rodilla - talon
    v1 = tuple(self.pose[0:2, 7].astype(np.int32))
    v2 = tuple(self.pose[0:2, 8].astype(np.int32))

    teta_3 = angulo_vertical(v1, v2)

    return [{'value': teta_1, 'title': 'Pecho-Cad'}, {'value': teta_2, 'title': 'Cad-Rod'}, {'value': teta_3, 'title': 'Rod-Talon'}]


def flex_brazo(self):

    # Izquierdo
    p1 = np.array(self.pose[0:2, 4].astype(np.int32))
    p2 = np.array(self.pose[0:2, 3].astype(np.int32))
    p3 = np.array(self.pose[0:2, 5].astype(np.int32))

    teta_i = ang_tresp(p1, p2, p3)

    # Derecho
    p1 = np.array(self.pose[0:2, 10].astype(np.int32))
    p2 = np.array(self.pose[0:2, 11].astype(np.int32))
    p3 = np.array(self.pose[0:2, 9].astype(np.int32))

    teta_d = ang_tresp(p1, p2, p3)

    return [{'value': teta_i, 'title': 'Flexion Izq.'}, {'value': teta_d, 'title': 'Flexion Der.'}]


def flex_cuello(self):

    np.seterr(divide='raise')

    v1 = tuple(self.pose[0:2, 18].astype(np.int32))
    v2 = tuple(self.pose[0:2, 17].astype(np.int32))
    try:
        teta = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
    except:
        if v2[0] == v1[0]:
            teta = 90
    return [{'value': teta, 'title': 'Inclinacion'}]


def cuello_post(self):

    # Desde la izquierda

    np.seterr(divide='raise')

    v1 = tuple(self.pose[0:2, 1].astype(np.int32))
    v2 = tuple(self.pose[0:2, 0].astype(np.int32))
    try:
        teta = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
    except:
        if v2[0] == v1[0]:
            teta = 90
    return [{'value': teta, 'title': 'Flexion Cuello'}]


def cuadriceps_der(self):

    np.seterr(divide='raise')

    # Vista desde la Derecha
    # Hombro - cadera
    v1 = tuple(self.pose[0:2, 0].astype(np.int32))
    v2 = tuple(self.pose[0:2, 12].astype(np.int32))

    teta_1 = angulo_vertical(v1, v2)

    #Cadera - rodilla
    v1 = tuple(self.pose[0:2, 12].astype(np.int32))
    v2 = tuple(self.pose[0:2, 13].astype(np.int32))

    teta_2 = angulo_vertical(v1, v2)

    # Derecha
    p1 = np.array(self.pose[0:2, 13].astype(np.int32))
    p2 = np.array(self.pose[0:2, 14].astype(np.int32))
    p3 = np.array(self.pose[0:2, 12].astype(np.int32))

    teta_3 = ang_tresp(p1, p2, p3)

    return [{'value': teta_1, 'title': 'Pecho-Cad'}, {'value': teta_2, 'title': 'Cad-Rod'}, {'value': teta_3, 'title': 'Flexion Rod.'}]


def cuadriceps_izq(self):

    np.seterr(divide='raise')

    # Vista desde la Izquierda
    # Hombro - cadera
    v1 = tuple(self.pose[0:2, 0].astype(np.int32))
    v2 = tuple(self.pose[0:2, 6].astype(np.int32))

    teta_1 = angulo_vertical(v1, v2)

    #Cadera - rodilla
    v1 = tuple(self.pose[0:2, 6].astype(np.int32))
    v2 = tuple(self.pose[0:2, 7].astype(np.int32))

    teta_2 = angulo_vertical(v1, v2)

    # Izquierda
    p1 = np.array(self.pose[0:2, 7].astype(np.int32))
    p2 = np.array(self.pose[0:2, 8].astype(np.int32))
    p3 = np.array(self.pose[0:2, 6].astype(np.int32))

    teta_3 = ang_tresp(p1, p2, p3)

    return [{'value': teta_1, 'title': 'Pecho-Cad'}, {'value': teta_2, 'title': 'Cad-Rod'}, {'value': teta_3, 'title': 'Flexion Rod.'}]


def isquiotibial_der(self):

    np.seterr(divide='raise')

    # Vista desde la Izquierda
    # Cadera - Rodilla
    v1 = tuple(self.pose[0:2, 12].astype(np.int32))
    v2 = tuple(self.pose[0:2, 13].astype(np.int32))

    try:
        teta_1 = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
    except:
        if v2[0] == v1[0]:
            teta_1 = 90
    if v2[0] < v1[0]:
        90+teta_1

    # Rodilla - Talon
    v1 = tuple(self.pose[0:2, 12].astype(np.int32))
    v2 = tuple(self.pose[0:2, 14].astype(np.int32))

    try:
        teta_2 = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
    except:
        if v2[0] == v1[0]:
            teta_2 = 90
    if v2[0] < v1[0]:
        90+teta_2

    return [{'value': teta_1, 'title': 'Cad-Rod'}, {'value': teta_2, 'title': 'Rod-Talon'}]


def isquiotibial_izq(self):

    np.seterr(divide='raise')

    # Vista desde la Izquierda
    # Cadera - Rodilla
    v1 = tuple(self.pose[0:2, 6].astype(np.int32))
    v2 = tuple(self.pose[0:2, 7].astype(np.int32))

    try:
        teta_1 = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
    except:
        if v2[0] == v1[0]:
            teta_1 = 90
    if v2[0] < v1[0]:
        90+teta_1

    # Rodilla - Talon
    v1 = tuple(self.pose[0:2, 6].astype(np.int32))
    v2 = tuple(self.pose[0:2, 8].astype(np.int32))

    try:
        teta_2 = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
    except:
        if v2[0] == v1[0]:
            teta_2 = 90
    if v2[0] < v1[0]:
        90+teta_2

    return [{'value': teta_1, 'title': 'Cad-Rod'}, {'value': teta_2, 'title': 'Rod-Talon'}]


def gluteo_der(self):

    np.seterr(divide='raise')

    # Angulo cadera
    p1 = np.array(self.pose[0:2, 12].astype(np.int32))
    p2 = np.array(self.pose[0:2, 13].astype(np.int32))
    p3 = np.array(self.pose[0:2, 0].astype(np.int32))

    teta_1 = ang_tresp(p1, p2, p3)

    # Angulo rodilla
    p1 = np.array(self.pose[0:2, 13].astype(np.int32))
    p2 = np.array(self.pose[0:2, 12].astype(np.int32))
    p3 = np.array(self.pose[0:2, 14].astype(np.int32))

    teta_2 = ang_tresp(p1, p2, p3)

    return [{'value': teta_2, 'title': 'Rodilla D.'}, {'value': teta_1, 'title': 'Cadera D.'}]


def gluteo_izq(self):
    np.seterr(divide='raise')

    # Angulo cadera

    p1 = np.array(self.pose[0:2, 6].astype(np.int32))
    p2 = np.array(self.pose[0:2, 0].astype(np.int32))
    p3 = np.array(self.pose[0:2, 7].astype(np.int32))

    teta_1 = ang_tresp(p1, p2, p3)

    # Angulo rodilla

    # Izquierda
    p1 = np.array(self.pose[0:2, 7].astype(np.int32))
    p2 = np.array(self.pose[0:2, 8].astype(np.int32))
    p3 = np.array(self.pose[0:2, 6].astype(np.int32))

    teta_2 = ang_tresp(p1, p2, p3)

    return [{'value': teta_2, 'title': 'Rodilla I.'}, {'value': teta_1, 'title': 'Cadera I.'}]


def circulos_cuello(self):

    np.seterr(divide='raise')

    v1 = tuple(self.pose[0:2, 0].astype(np.int32))
    v2 = tuple(self.pose[0:2, 1].astype(np.int32))
    try:
        teta_1 = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
    except:
        if v2[0] == v1[0]:
            teta_1 = 90
    if v1[0] >= v2[0]:
        teta_1 = 180-teta_1

    v1 = tuple(self.pose[0:2, 18].astype(np.int32))
    v2 = tuple(self.pose[0:2, 17].astype(np.int32))
    try:
        teta_2 = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
    except:
        if v2[0] == v1[0]:
            teta_2 = 90

    return [{'value': teta_1, 'title': 'Incl. Cuello'}, {'value': teta_2, 'title': 'Incl. Cara'}]


def circulos_hombros(self):

    np.seterr(divide='raise')

    # Vista desde la Izquierda
    # Cadera - Rodilla
    v1 = tuple(self.pose[0:2, 1].astype(np.int32))
    v2 = tuple(self.pose[0:2, 9].astype(np.int32))

    try:
        teta_1 = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
    except:
        if v2[0] == v1[0]:
            teta_1 = 0
    if v2[0] < v1[0]:
        90+teta_1

    # Rodilla - Talon
    v1 = tuple(self.pose[0:2, 1].astype(np.int32))
    v2 = tuple(self.pose[0:2, 3].astype(np.int32))

    try:
        teta_2 = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
    except:
        if v2[0] == v1[0]:
            teta_2 = 0
    if v2[0] < v1[0]:
        90+teta_2

    return [{'value': teta_1, 'title': 'Hombro Der.'}, {'value': teta_2, 'title': 'Hombro Izq.'}]


def circulos_caderas(self):

    np.seterr(divide='raise')

    p1 = tuple(self.pose[0:2, 14].astype(np.int32))
    p2 = tuple(self.pose[0:2, 8].astype(np.int32))

    v1 = tuple([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2])

    p1 = tuple(self.pose[0:2, 12].astype(np.int32))
    p2 = tuple(self.pose[0:2, 6].astype(np.int32))

    v2 = tuple([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2])

    # Vista desde la Izquierda
    # Hombro - cadera

    try:
        teta = math.degrees(math.atan(abs(v2[0]-v1[0])/abs(v2[1]-v1[1])))
    except:
        if v2[1] == v1[1]:
            teta = 90

    if v2[0] <= v1[0]:
        teta += 90
    else:
        teta = 90-teta

    return [{'value': teta, 'title': 'Cadera'}]


def circulos_rodillas(self):

    np.seterr(divide='raise')

    p1 = tuple(self.pose[0:2, 14].astype(np.int32))
    p2 = tuple(self.pose[0:2, 8].astype(np.int32))

    v1 = tuple([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2])

    p1 = tuple(self.pose[0:2, 13].astype(np.int32))
    p2 = tuple(self.pose[0:2, 7].astype(np.int32))

    v2 = tuple([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2])

    # Vista desde la Izquierda
    # Hombro - cadera

    try:
        teta = math.degrees(math.atan(abs(v2[0]-v1[0])/abs(v2[1]-v1[1])))
    except:
        if v2[1] == v1[1]:
            teta = 90

    if v2[0] <= v1[0]:
        teta += 90
    else:
        teta = 90-teta

    return [{'value': teta, 'title': 'Rodillas'}]

def bike_lateral(self):
    if(self.side):
        # Brazo
        p1 = np.array(self.pose[0:2, 9])
        p2 = np.array(self.pose[0:2, 12])
        p3 = np.array(self.pose[0:2, 10])
        teta_b = ang_tresp(p1,p2,p3)

        # Cadera
        v1 = tuple(self.pose[0:2, 12])
        v2 = tuple(self.pose[0:2, 9])
        try:
            teta_c = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
        except:
            if v2[0] == v1[0]:
                teta_c = 90

        # Rodilla
        p1 = np.array(self.pose[0:2, 13])
        p2 = np.array(self.pose[0:2, 12])
        p3 = np.array(self.pose[0:2, 14])
        teta_r = ang_tresp(p1,p2,p3)
    else:
        # Brazo
        p1 = np.array(self.pose[0:2, 3])
        p2 = np.array(self.pose[0:2, 4])
        p3 = np.array(self.pose[0:2, 6])
        teta_b = ang_tresp(p1,p2,p3)

        # Cadera
        v1 = tuple(self.pose[0:2, 6])
        v2 = tuple(self.pose[0:2, 3])
        try:
            teta_c = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
        except:
            if v2[0] == v1[0]:
                teta_c = 90

        # Rodilla
        p1 = np.array(self.pose[0:2, 7])
        p2 = np.array(self.pose[0:2, 8])
        p3 = np.array(self.pose[0:2, 6])
        teta_r = ang_tresp(p1,p2,p3)


    return [{'value':teta_b,'title':'Brazo'},{'value':teta_c,'title':'Cadera'},{'value':teta_r,'title':'Rodilla'}]

def bike_post(self):

    # Rodilla
    p1 = np.array(self.pose[0:2, 13])
    p2 = np.array(self.pose[0:2, 12])
    p3 = np.array(self.pose[0:2, 14])
    teta_rd = ang_tresp(p1,p2,p3)

    # Rodilla
    p1 = np.array(self.pose[0:2, 7])
    p2 = np.array(self.pose[0:2, 8])
    p3 = np.array(self.pose[0:2, 6])
    teta_ri = ang_tresp(p1,p2,p3)

    return [{'value':teta_rd,'title':'Rodilla Der'},{'value':teta_ri,'title':'Rodilla Izq'}]


class Exercises:
    angle_calculation = {'C1': cadera_ext,
                         'C2': cadera_flex,
                         'C3': cadera_rot,
                         'R1': rodilla_flex,
                         'CR1': craneo_cerv_flex,
                         'CR2': craneo_cerv_incl,
                         'H1': hombro_flex,
                         'H2': hombro_abdu,
                         'H3': hombro_ext,
                         'H4': hombro_rot_int,
                         'H5': hombro_rot_ext,
                         'CO1': codo_flex,
                         'D_SH': sep_hombros,
                         'D_ER': elev_rodillas,
                         'D_TG': talon_gluteo,
                         'D_PS': parar_sentar,
                         'D_S': sentadilla,
                         'D_FB': flex_brazo,
                         'E_FC': flex_cuello,
                         'E_CP': cuello_post,
                         'E_CD': cuadriceps_der,
                         'E_CI': cuadriceps_izq,
                         'E_ID': isquiotibial_der,
                         'E_II': isquiotibial_izq,
                         'E_GD': gluteo_der,
                         'E_GI': gluteo_izq,
                         'M_CU': circulos_cuello,
                         'M_CH': circulos_hombros,
                         'M_CA': circulos_caderas,
                         'M_CR': circulos_rodillas,

                         'BF_LAT': bike_lateral,
                         'BF_POS': bike_post
                         }

    def __init__(self, tag, pose, side):
        self.tag = tag
        self.pose = pose
        self.side = side

    def calculate(self):
        # np.seterr(divide='raise')
        if self.tag not in self.angle_calculation:
            return None
        return self.angle_calculation[self.tag](self)
