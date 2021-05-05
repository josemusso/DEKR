# -*- coding: utf-8 -*-
import math
import numpy as np
""" 
================================================================================================
    Support Functions
================================================================================================
"""

def horizontal_angle(self,a,b):
    """ Measures the angle formed by two points using a horizontal line drawed from point a."""
    v1 = tuple(self.pose[0:2, a])
    v2 = tuple(self.pose[0:2, b])
    try:
        teta = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
    except:
        if v2[0] == v1[0]:
            teta = 90
    return teta

def vertical_angle(self, a,b):
    """ Measures the angle formed by two points using a vertical line drawed from point a."""
    v1 = tuple(self.pose[0:2, a])
    v2 = tuple(self.pose[0:2, b])
    try:
        teta = math.degrees(math.atan(abs(v2[0]-v1[0])/abs(v2[1]-v1[1])))
    except:
        if v2[1] == v1[1]:
            teta = 90

    return teta

def ang_threep(a,b,c):
    """
    Measures the angle formed by three points. The angle goes from 0 to 360. 
    Point 'a' it's the anchord, and the measured angle it's the one that goes from 'b' to 'c' clockwise.
    """
    ang = math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]) - math.atan2(c[1]-a[1], c[0]-a[0]))
    return ang + 360 if ang < 0 else ang

def angle_measure(self,a,b,c):
    """
    Gets keypoint data using the numbers provided and then calls "ang_threep()" to calculate the angle.
    """
    p1 = np.array(self.pose[0:2, a])
    p2 = np.array(self.pose[0:2, b])
    p3 = np.array(self.pose[0:2, c])
    
    angle = ang_threep(p1,p2,p3)
    return angle

""" 
================================================================================================
    Exercises
================================================================================================
"""
def cuello_lateral(self):
    """
    This angle describes the head inclination. 
    When the head it's on the center angle=0, while moving to the right the angle it's negative and when it's to the left it's positive.
    The angles used are formed by both shoulders and each ear, after substracting the one formed by the right ear by the left one, we get the result.
    """
    # Neck inclination
    ## Left
    teta_i=angle_measure(self,9,3,18)
    ## Right
    teta_d = angle_measure(self,3,17,9)

    teta = teta_d - teta_i

    return [{'value':teta,'title':'Inclination'}]


def cuello_posterior(self):
    """
    To measure head forward inclination we dont have 3 points, so we only use the nose and the shouder closer to the camera.
    The angle it's the one formed by a vertical line that comes from the shoulder and the nose.
    """
    np.seterr(divide='raise')
    if(self.side):
        # Right
        v1 = tuple(self.pose[0:2, 1])
        v2 = tuple(self.pose[0:2, 9])
        try:
            teta = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
        except:
            if v2[0] == v1[0]:
                teta = 90
    else:
        # Left

        v1 = tuple(self.pose[0:2, 1])
        v2 = tuple(self.pose[0:2, 3])
        try:
            teta = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
        except:
            if v2[0] == v1[0]:
                teta = 90
    return [{'value':teta,'title':'Neck Flexion'}]

def circunduccion_hombro_izquierdo(self):
    """
    Measures the circular movement done by the left arm.
    Used points: Shoulder, ear, hand.
    The reason to use the ear instead of the hip as a reference point is that the rest position of the arm is near the hip 
    so it's better to have the 'start' and 'end' of the movement on other point.
    """
    teta=angle_measure(self,3,18,4)
 
    return [{'value':teta,'title':'Left arm'}]

def circunduccion_hombro_derecho(self):
    """
    Measures the circular movement done by the right arm.
    Used points: Shoulder, ear, hand.
    The reason to use the ear instead of the hip as a reference point it's that the rest position of the arm it's near the hip 
    so it's better to have the 'start' and 'end' of the movement on other point.
    """
    teta=angle_measure(self,9,17,11)

    return [{'value':teta,'title':'Right arm.'}]

def sep_brazos(self):
    """
    Measures the elevation of both arms.
    Used points: Shoulder, Hip, hand. Each side.
    """
    ## Left
    teta_i = angle_measure(self,3,6,5)
    ## Right
    teta_d = angle_measure(self,9,11,12)

    return [{'value':teta_i,'title':'Left arm.'},{'value':teta_d,'title':'Right arm'}]

def circunduccion_codos(self):
    """
    Measures the circular movement of both arms by using flexion.
    Used points: Elbow, hand, shoulder. Each side.
    """
    ## Left
    teta_i = angle_measure(self,4,5,3)
    ## Right
    teta_d = angle_measure(self,10,9,11)

    return [{'value':teta_i,'title':'Flexion Izq.'},{'value':teta_d,'title':'Flexion Der.'}]

def lateralizacion_columna(self):
    """
    Measures the elevation of both arms and hip extension on both sides.
    Used points: Shoulder, Hip, hand, each side for arms and hip, ankle, shoulder, each side for body inclination.
    """
    ## Left
    # Arm
    teta_ib = angle_measure(self,3,6,5)
    # Hip
    teta_ic = angle_measure(self,6,8,3)

    ## Right
    # Arm
    teta_db = angle_measure(self,9,11,12)
    # Hip
    teta_dc = angle_measure(self,12,9,14)

    return [{'value':teta_ib,'title':'Left Arm.'},{'value':teta_ic,'title':'Left hip'},{'value':teta_db,'title':'Right arm'},{'value':teta_dc,'title':'Right hip'}]

def rotacion_columna(self):
    """
    Estimates rotation of the body by measuring the angle formed by the nose and the shoulders.
    Whem the user rotates the angle goes to 0. This approach doesn't give direction of the rotation.
    Used points: Nose, r. shoulder, l. shoulder.
    """
    #Pecho
    p2 = np.array(self.pose[0:2, 9])
    p3 = np.array(self.pose[0:2, 3])
 
    if (p2[0]>p3[0]):
        teta = angle_measure(self,1,3,9)
    else:
        teta = angle_measure(self,1,9,3)

    if (teta>=200):
        teta=0
    
    return [{'value':teta,'title':'Rotation'}]


def circunduccion_caderas(self):
    """
    Estimates rotation of the hips by measuring the side to side movement.
    There are two parts for this measure. For each side first we get the difference between hip extension and flexion angles (inside and outside of the body).
    Then the difference from both sides it's added to result on one angle that describes lateral movement. 
    When the person is on neutral position angle=0, while moving to the right angle it's positive and to the left it's negative.
    Used points: Shoulder, Hip, Knee, each side.
    """
    # Right
    teta_1 = angle_measure(self,12,9,13)
    teta_2 = angle_measure(self,12,13,9)
    teta_d = teta_1 - teta_2
    # Left
    teta_1 = angle_measure(self,6,3,7)
    teta_2 = angle_measure(self,6,7,3)
    teta_i = teta_1 - teta_2

    # Hip displacement
    teta = teta_d + teta_i

    return [{'value':teta,'title':'Hip displacement'}]

def circunduccion_rodillas(self):
    """
    Estimates rotation of the knees by measuring the side to side movement.
    There are two parts for this measure. For each side first we get the difference between knee flexion seen to the inside and outside of the body.
    Then the difference from both sides it's added to result on one angle that describes lateral movement. 
    When the person is on neutral position angle=0, while moving to the right angle it's positive and to the left it's negative.
    Used points: Shoulder, Hip, Knee, each side.
    """
    #Right
    teta_1 = angle_measure(self,13,12,14)
    teta_2 = angle_measure(self,13,14,12)
    teta_d = teta_1 - teta_2
    #Left
    teta_1 = angle_measure(self,7,6,8)
    teta_2 = angle_measure(self,7,8,6)
    teta_i = teta_1 - teta_2

    #Knee displacement
    teta = teta_d + teta_i

    return [{'value':teta,'title':'Knee displacement'}]

def pecho(self):
    """
    Measures the elevation of both arms and position of the body (column has to be straight).
    The poits used deppend on the side given to the cammera.
    Used points: Shoulder, Hip, hand, for arms and hip, ankle, shoulder for body inclination.
    """
    if (self.side):
        # Right side
        # Arm
        teta_1 = angle_measure(self,9,11,12)
        # Column
        teta_2 = angle_measure(self,12,14,9)
    else:
        # Left side
        # Arm
        teta_1 = angle_measure(self,3,6,5)
        # Column
        teta_2 = angle_measure(self,12,8,3)

    return [{'value':teta_1,'title':'Arm'},{'value':teta_2,'title':'Column'}]

def inclinacion_columna_izq(self):
    """
    Measures the elevation of left arm and hip extension.
    Used points: Shoulder, Hip, hand for arm and hip, ankle, shoulder for body inclination.
    """
    # Arm
    teta_db = angle_measure(self,9,11,12)
    # Hip
    teta_dc = angle_measure(self,12,9,14)

    return [{'value':teta_db,'title':'Right arm.'},{'value':teta_dc,'title':'Right hip.'}]


def inclinacion_columna_der(self):
    """
    Measures the elevation of right arm and hip extension.
    Used points: Shoulder, Hip, hand for arm and hip, ankle, shoulder for body inclination.
    """
    # Arm
    teta_ib = angle_measure(self,3,6,5)
    # Hip
    teta_ic = angle_measure(self,6,8,3)

    return [{'value':teta_ib,'title':'Left arm'},{'value':teta_ic,'title':'Left hip'}]

def flexion_columna(self):
    """
    Measures hip flexion from lateral view.
    Used points: hip, ankle, shoulder.
    """
    if(self.side):
        #Right side
        teta = angle_measure(self,12,14,9)
    else:
        #Left side
        teta = angle_measure(self,12,3,8)

    return [{'value':teta,'title':'Hip'}]

def gluteo_der(self):
    """
    Estimates if the right leg it's on the correct position. This means being over the left knee.
    By measuring the angle formed by the right hip, left knee and right ankle, when the angle it's "low", it means the leg it's on the right position.
    Used points: r. hip, l. knee, r. ankle.
    """
    # Leg position
    teta = angle_measure(self,12,7,14)

    return [{'value':teta,'title':'Leg position'}]


def gluteo_izq(self):
    """
    Estimates if the left leg it's on the correct position. This means being over the right knee.
    By measuring the angle formed by the left hip, right knee and left ankle, when the angle it's "low", it means the leg it's on the right position.
    Used points: l. hip, r. knee, l. ankle.
    """
    ## Leg position
    teta = angle_measure(self,6,8,13)

    return [{'value':teta,'title':'Leg position'}]

def gemelos_izq(self):
    """
    Estimates inclination by using the angle formed by the shoulder and both ankles. Also measures hip and knee fletion.
    Used points: Shoulder, ankles for inclination. Hip, shoulder, knee for hip flexion. Knee hip and ankle for knee flexion.
    """
    # Hip
    teta_1 = angle_measure(self,6,3,7)
    # Knee
    teta_2 = angle_measure(self,7,6,8)
    # Inclination
    teta_3 = angle_measure(self,3,8,14)

    return [{'value':teta_3,'title':'Inclination'},{'value':teta_1,'title':'Hip'},{'value':teta_2,'title':'Knee'}]

def gemelos_der(self):
    """
    Estimates inclination by using the angle formed by the shoulder and both ankles. Also measures hip and knee fletion.
    Used points: Shoulder, ankles for inclination. Hip, shoulder, knee for hip flexion. Knee hip and ankle for knee flexion.
    """
    # Hip
    teta_1 = angle_measure(self,12,9,13)
    # Knee
    teta_2 = angle_measure(self,13,12,14)
    # Inclination
    teta_3 = angle_measure(self,9,14,8)

    return [{'value':teta_3,'title':'Inclination'},{'value':teta_1,'title':'Hip'},{'value':teta_2,'title':'Knee'}]


def marcha_estacionaria(self):
    """
    Measures hip and knee flexion (the side given to the cammera)
    Used points: Hip, shoulder, knee for hip flexion. Knee, hip and ankle for knee flexion.
    """
    if(self.side):
        # Right side
        # Hip
        teta_d1 = angle_measure(self,12,13,9)
        # Knee
        teta_d2 = angle_measure(self,13,12,14)
    else:
        #Left side
        # Hip
        teta_d1 =angle_measure(self,6,3,7)
        # Knee
        teta_d2 = angle_measure(self,7,8,6)
    return [{'value':teta_d2,'title':'Knee'},{'value':teta_d1,'title':'Hip'}]

def talon_gluteo(self):
    """
    Measures knee flexion (the side given to the cammera)
    Used points: Knee, hip and ankle for knee flexion.
    """
    if(self.side):
        ## Right side
        teta_d = angle_measure(self,13,12,14)
    else:
        ## Left side
        teta_d = angle_measure(self,7,8,6)

    return [{'value':teta_d,'title':'Knee'}]



def parar_sentar(self):
    """
    Measures hip flexion from two points, one using shoulder and knee and other using shoulder and ankle (the side given to the cammera).
    This way we make sure the peson asumes the position.
    Used points: Hip, shoulder, knee. Hip, shoulder, ankle.
    """

    if(self.side):
        # Right side
        ## Hip
        teta_1 = angle_measure(self,12,13,9)
        ## Knee
        teta_2 = angle_measure(self,12,14,9)
    else:
        # Left side
        ## Hip
        teta_1 = angle_measure(self,6,3,7)
        ##Knee
        teta_2 = angle_measure(self,6,3,8)

    return [{'value':teta_2,'title':'Hip'},{'value':teta_1,'title':'Column'}]


def sentadilla_lat(self):
    """
    Estimates body displacement by measuring the angle formed by the center of the chest and knees.
    When the person opens his legs and goes down this angle gets bigger.
    Used points: chest, r. knee, l. knee.
    """
    teta_1 = angle_measure(self,0,13,7)

    return [{'value':teta_1,'title':'Knees'}]

def rodilla_codo(self):
    """
    Estimates how close the elbow is to the opposing knee by measuring the angle formed from the hip (the one that's on the same side of the elbow).
    When the person tocheshis knee with his elbow the angle gets smaller.
    Used points: hip, knee, elbow for each side.
    """
    ## R. elbow - L. knee
    teta_1 = angle_measure(self,12,7,10)
    ## L. Elbow - R. knee
    teta_2 = angle_measure(self,6,4,13)

    return [{'value':teta_1,'title':'RE-LK'},{'value':teta_2,'title':'LE-RK'}]

def brincos_jack(self):
    """
    Measures elevation of the arms and estimates body displacement.
    Both arms are represented with only one angle, the sum of each arm elevation. 
    As the elevation might be on the same axis as the body, it's measured from the shoulder opposed to the arm.
    For the body, the angle measured is the one formmed by the chest and both ankles.
    Used points: shoulder, hand, hip for arms. Chest and ankles for the body.
    """
    ## Arms
    # Right
    teta_d = angle_measure(self,3,11,12)
    # Left
    teta_i = angle_measure(self,9,6,5)
    teta_1 = teta_d+teta_i

    ## Legs
    teta_2 = angle_measure(self,0,14,8)

    return [{'value':teta_1,'title':'Arms'},{'value':teta_2,'title':'Legs'}]

def paso_adelante_atras(self):
    """
    Estimates side to side movement by just measuring the angle formmed by the hip (closest to the cammera) and both ankles.
    The measurement changes depending on wich ankle it's in front so we always get the inside angle.
    There's no direction of the movement.
    Used points: hip, both ankles.
    """
    if(self.side):
        # Right side
        p2 = np.array(self.pose[0:2, 8])
        p3 = np.array(self.pose[0:2, 14])
        if(p3[0]>p2[0]):
            teta = angle_measure(self,12,8,14)
        else:
            teta = angle_measure(self,12,14,18)

        if(teta>=100 or teta<=-100):
            teta=0
    else:
        # Left side
        p2 = np.array(self.pose[0:2, 8])
        p3 = np.array(self.pose[0:2, 14])
        if(p3[0]>p2[0]):
            teta = angle_measure(self,6,8,14)
        else:
            teta = angle_measure(self,6,14,8)

        if(teta>=100 or teta<=-100):
            teta=0

    return [{'value':teta,'title':'Legs'}]

def boxing(self):
    """
    Estimates arm extension.
    Measures the angle formed by the hip, the shoulder and the hand (closest to the camera).
    Used points: hip, hand, shoulder.
    """
    if(self.side):
        # Right side
        teta = angle_measure(self,12,11,9)
    else:
        # Left side
        teta = angle_measure(self,6,3,5)
    return [{'value':teta,'title':'Arm'}]

def saltar_cuerda(self):
    """
    Measures arm and knee flexion (closest to the cammera).
    Used points: knee, hip, ankle for knee. Elbow, shoulder, hand for arm.
    """
    if(self.side):
        # Right side
        ## Knee
        teta_r = angle_measure(self,13,12,14)
        ## Elbow
        teta_c = angle_measure(self,10,11,9)
    else:
        # Left side
        ## Knee
        teta_r = angle_measure(self,7,8,6)
        ## Elbow
        teta_c = angle_measure(self,4,3,5)

    return [{'value':teta_r,'title':'Knee'},{'value':teta_c,'title':'Elbow'}]

def rotacion_cuello(self):
    """
    This angle describes the head rotation. 
    When the head it's on the center angle=0, while moving to the right the angle it's negative and when it's to the left it's positive.
    The angles used are formed by both shoulders and the nose, after substracting both, we get the result.
    """
    ## Left
    teta_i = angle_measure(self,9,3,1)
    ## Right
    teta_d = angle_measure(self,3,1,9)

    # Head Rotation
    teta = teta_d - teta_i

    return [{'value':teta,'title':'Rotation'}]

def triceps(self):
    """
    Measures arm flexion (closest to the cammera).
    Used points: Elbow, shoulder, hand for arm.
    """
    if(self.side):
        ## Right side
        teta = angle_measure(self,10,11,9)
    else:
        ## Left side
        teta = angle_measure(self,4,3,5)

    return [{'value':teta,'title':'Elbow'}]

def aplauso(self):
    """
    Estimates arm displacement (closest to the cammera).
    measures the angle formed bi the shoulder, hand and hip. If the hand it's in front of the body the angle is positive, when it's behind the angle it's negative.
    Used points: shoulder, hip, hand.
    """
    if(self.side):
        ## Right side
        p1 = np.array(self.pose[0:2, 9])
        p3 = np.array(self.pose[0:2, 11]) 
        if(p3[0]>p1[0]):
            teta = -angle_measure(self,9,12,11)
        else:
            teta = angle_measure(self,9,11,12)

    else:
        # Left side
        p1 = np.array(self.pose[0:2, 3])
        p2 = np.array(self.pose[0:2, 5])
        if(p2[0]<p1[0]):
            teta = -angle_measure(self,3,5,6)
        else:
            teta = angle_measure(self,3,6,5)

    return [{'value':teta,'title':' Arm'}]

def flexoextension_columna(self):
    """
    Measures hip flexion/extension seen from the side.
    Used points: hip, knee, shoulder.
    """
    if(self.side):
        # Right side
        teta = angle_measure(self,12,13,9)
    else:
        #Left side
        teta = angle_measure(self,6,3,7)

    return [{'value':teta,'title':'Hip'}]

def inclinacion_flexion(self):
    """
    Estimates body displacement.
    Measures the angle formed by each hip with both ankles and then getting the difference between the right hip and left hip.
    When moving to the left, angle it's positive and gets negative when moving to the right.
    Used points: shoulder, hip, hand.
    """
    ## R. hip
    teta_1 = angle_measure(self,12,14,8)
    ## L. hip
    teta_2 = angle_measure(self,6,14,8)

    teta=teta_1-teta_2

    return [{'value':teta,'title':'Movement'}]

def brazo_pierna(self):
    """
    Measures elevation of the arms and estimates leg position.
    Both arms are represented with only one angle, the difference of each arm elevation, right minus left.
    Both legs are represented with only one angle, the difference of each ankle displacement.
    Leg position is measured using shoulder, ankle and the oposing side hip.
    Used points: shoulder, ankle, opposing hip for legs. shoulder, hand, hip for arms.
    """
    ## Legs
    #Right
    teta_1 = angle_measure(self,9,14,6)
    #Left
    teta_2 = angle_measure(self,3,12,8)

    teta_p=teta_1-teta_2

    ## Arms
    #Right
    teta_1 = angle_measure(self,9,11,12)
    #Left
    teta_2 = angle_measure(self,3,6,5)

    teta_b=teta_1-teta_2

    return [{'value':teta_b,'title':'Arms'},{'value':teta_p,'title':'Legs'}]

def mano_talon(self):
    """
    Estimates how close the hand is to the opposing ankle by measuring the angle formed from the hip (the one that's on the same side of the ankle).
    When the person touches his ankle with his hand the angle gets smaller.
    Used points: hip, ankle, hand for each side.
    """
    ## right hand - left feet
    teta_1 = angle_measure(self,6,11,8)
    ## Left hand - right feet
    teta_2 = angle_measure(self,12,14,5)

    return [{'value':teta_1,'title':'RH-LF'},{'value':teta_2,'title':'LH-RF'}]

def flexbrazo_extcadera(self):
    """
    Measures elevation of the arm and hip flexion (closest to cammera).
    Used points: shoulder, hip, hand for the arm. Hip, ankle, shoulder for hip.
    """
    if(self.side):
        #Right side
        ## Arm
        teta_b = angle_measure(self,9,12,11)
        ## Hip
        teta_c = angle_measure(self,12,14,9)
    else:
        # Left side
        ## Arm
        teta_b = angle_measure(self,3,5,6)
        ## Hip
        teta_c = angle_measure(self,6,3,8)

    return [{'value':teta_b,'title':'Arm'},{'value':teta_c,'title':'Hip'}]

def sentadilla(self):
    """
    Measures knee and hip flexion (closest to cammera).
    Used points: knee, hip, ankle for the arm. Hip, ankle, shoulder for hip.
    """
    if(self.side):
        # Right side
        # Hip
        teta_1 = angle_measure(self,12,13,9)
        # Knee
        teta_2 = angle_measure(self,13,12,14)
    else:
        # Left side
        # Hip
        teta_1 = angle_measure(self,6,3,7)
        # Knee
        teta_2 = angle_measure(self,7,8,6)

    return [{'value':teta_2,'title':'Knee'},{'value':teta_1,'title':'Hip'}]

def apoyo_unipodal_der(self):
    """
    Measures right knee and hip flexion .
    Used points: knee, hip, ankle for the arm. Hip, ankle, shoulder for hip.
    """
    # Hip
    teta_1 = angle_measure(self,6,3,7)
    # Knee
    teta_2 = angle_measure(self,7,8,6)

    return [{'value':teta_2,'title':'Knee'},{'value':teta_1,'title':'Hip'}]


def apoyo_unipodal_izq(self):
    """
    Measures left knee and hip flexion.
    Used points: knee, hip, ankle for the arm. Hip, ankle, shoulder for hip.
    """
    # Hip
    teta_1 = angle_measure(self,12,13,9)
    # Knee
    teta_2 = angle_measure(self,13,12,14)

    return [{'value':teta_2,'title':'Knee'},{'value':teta_1,'title':'Hip'}]

def squat_front(self):
    """
    Test for other product
    """
    # P. derecha
    teta_pd = vertical_angle(self,13,14)

    # P. Izquierda
    teta_pi = vertical_angle(self,8,7)

    return [{'value':teta_pd,'title':'P. Der'},{'value':teta_pi,'title':'P. Izq'}]

def squat_lat(self):
    """
    Test for other product
    """

    if(self.side):
        # Cadera
        teta_c = angle_measure(self,12,13,0)

        # Rodilla
        teta_r = angle_measure(self,13,12,14)

        #Alineado
        teta_a= teta_c-teta_r

        # Muslo
        teta_m = horizontal_angle(self, 12, 13)

        # Vara
        v  = tuple(self.pose[0:2, 11])
        p1 = tuple(self.pose[0:2, 20])
        p2 = tuple(self.pose[0:2, 21])
        #pm = (p1[0]+p2[0])/2

        if(v[0]<p1[0] or v[0]>p2[0]):
            teta_v=0
        else:
            teta_v=10
    else:
        # Cadera
        teta_c = angle_measure(self,6,0,7)

        # Rodilla
        teta_r = angle_measure(self,7,8,6)

        #Alineado
        teta_a= teta_c-teta_r

        # Muslo
        teta_m = horizontal_angle(self, 6, 7)

        # Vara
        v  = tuple(self.pose[0:2, 5])
        p1 = tuple(self.pose[0:2, 22])
        p2 = tuple(self.pose[0:2, 23])
        #pm = (p1[0]+p2[0])/2

        if(v[0]>p1[0] or v[0]<p2[0]):
            teta_v=0
        else:
            teta_v=10

    return [{'value':teta_a,'title':'Tronco-Tibia'},{'value':teta_m,'title':'Muslo'},{'value':teta_v,'title':'Vara'}]

def est_der_front(self):
    """
    Test for other product
    """
    # P. derecha
    v1 = tuple(self.pose[0:2, 13])
    v2 = tuple(self.pose[0:2, 14])
    try:
        teta_pd = math.degrees(math.atan(abs(v2[0]-v1[0])/abs(v2[1]-v1[1])))
    except:
        if v2[1] == v1[1]:
            teta_pd = 90

    # Pelvis
    v1 = tuple(self.pose[0:2, 12])
    v2 = tuple(self.pose[0:2, 6])
    try:
        teta_p = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
    except:
        if v2[0] == v1[0]:
            teta_p = 90
    
    # Hombros
    v1 = tuple(self.pose[0:2, 9])
    v2 = tuple(self.pose[0:2, 3])
    try:
        teta_h = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
    except:
        if v2[0] == v1[0]:
            teta_h = 90

    return [{'value':teta_pd,'title':'Pierna derecha'},{'value':teta_p,'title':'Pelvis'},{'value':teta_h,'title':'Hombros'}]

def est_der_lat(self):
    """
    Test for other product
    """
    # Cadera
    teta_c = angle_measure(self,12,13,9)

    # Rodilla
    teta_r = angle_measure(self,13,12,14)

    #Alineado
    teta_a= teta_c-teta_r

    # Muslo
    v1 = tuple(self.pose[0:2, 12])
    v2 = tuple(self.pose[0:2, 13])
    try:
        teta_m = math.degrees(math.atan(abs(v2[1]-v1[1])/abs(v2[0]-v1[0])))
    except:
        if v2[0] == v1[0]:
            teta_m = 90

    return [{'value':teta_a,'title':'Tronco-Tibia'},{'value':teta_m,'title':'Muslo'}]

def apoyo_brazo_pierna(self):
    """
    Test for other product
    """
    if(self.side):
        # Brazo
        teta_b = angle_measure(self,3,6,5)

        # Pierna
        teta_p = angle_measure(self,12,13,9)

        # Tronco- Pelvis
        teta_t=horizontal_angle(self,9,12)

        #Cadera
        teta_c = angle_measure(self,9,12,11)
    else:
        # Brazo
        teta_b = angle_measure(self,9,11,12)

        # pierna
        teta_p = angle_measure(self,6,3,7)

        # Tronco- Pelvis
        teta_t=horizontal_angle(self,3,6)
    
        #Cadera
        teta_c = angle_measure(self,3,5,6)

    return [{'value':teta_b,'title':'Brazo'},{'value':teta_p,'title':'Pierna'},{'value':teta_t,'title':'Tronco'},{'value':teta_c,'title':'Cadera'}]

def leg_rise_der(self):
    """
    Test for other product
    """
    # Cadera
    teta_p = 180-angle_measure(self,12,14,9)
    if(teta_p<0):
        teta_p=0

    return [{'value':teta_p,'title':'Pierna'}]

def leg_rise_izq(self):
    """
    Test for other product
    """
    # Cadera
    teta_p = 180-angle_measure(self,6,3,8)
    if(teta_p<0):
        teta_p=0

    return [{'value':teta_p,'title':'Pierna'}]



def puente_gluteo(self):

    if(self.side):
        teta_1 = angle_measure(self,9,11,12)
        teta_2 = angle_measure(self,12,13,9)
        teta_3 = angle_measure(self,13,12,14)
    else:
        teta_1 = angle_measure(self,3,6,5)
        teta_2 = angle_measure(self,6,3,7)
        teta_3 = angle_measure(self,7,8,6)

    return [{'value':teta_1,'title':'Elevacion'},{'value':teta_2,'title':'Cadera'},{'value':teta_3,'title':'Rodilla'}]

def plancha_lateral_izq(self):
    teta_2 = angle_measure(self,8,5,0)
    teta_3 = angle_measure(self,8,5,6)

    return [{'value':teta_2,'title':'pecho'},{'value':teta_3,'title':'Cadera'}]

def plancha_lateral_der(self):

    teta_2 = angle_measure(self,14,0,11)
    teta_3 = angle_measure(self,14,12,11)

    return [{'value':teta_2,'title':'pecho'},{'value':teta_3,'title':'Cadera'}]

def push_up(self):
    if(self.side):
        teta_1 = angle_measure(self,14,11,12)
        teta_2 = angle_measure(self,14,11,9)
        teta_3 = angle_measure(self,14,11,1)
    else:
        teta_1 = angle_measure(self,8,5,6)
        teta_2 = angle_measure(self,8,5,3)
        teta_3 = angle_measure(self,8,5,1)

    return [{'value':teta_1,'title':'cadera'},{'value':teta_2,'title':'pecho'},{'value':teta_3,'title':'cara'}]

def leg_squat_front_der(self):
    
    """
    Test for other product
    """
    # P. der
    teta_pd=vertical_angle(self, 13,14)

    # Pelvis
    teta_p=horizontal_angle(self, 12,6)
    
    # Hombros
    teta_h=horizontal_angle(self, 9,3)

    # Cuerpo
    teta_c=vertical_angle(self,14 , 0)

    return [{'value':teta_pd,'title':'Pierna derecha'},{'value':teta_p,'title':'Pelvis'},{'value':teta_h,'title':'Hombros'},{'value':teta_c,'title':'Cuerpo'}]

def leg_squat_front_izq(self):

    """
    Test for other product
    """
    # P. izq
    teta_pi=vertical_angle(self, 7,8)

    # Pelvis
    teta_p=horizontal_angle(self, 12,6)

    # Hombros
    teta_h=horizontal_angle(self, 9,3)

    # Cuerpo
    teta_c=vertical_angle(self,8 , 0)


    return [{'value':teta_pi,'title':'Pierna izquierda'},{'value':teta_p,'title':'Pelvis'},{'value':teta_h,'title':'Hombros'},{'value':teta_c,'title':'Cuerpo'}]

def leg_squat_lat(self):
    """
    Test for other product
    """
    if(self.side):
        teta = angle_measure(self,13,12,14)
    else:
        teta = angle_measure(self,7,8,6)
    return [{'value':teta,'title':'Rodilla'}]


def bike_lateral(self):
    """
    Test for other product
    """
    if(self.side):
        # Brazo
        teta_b = angle_measure(self,9,12,10)

        # Cadera
        teta_c=horizontal_angle(self,12,9)

        # Rodilla
        teta_r = angle_measure(self,13,12,14)
    else:
        # Brazo
        teta_b = angle_measure(self,3,4,6)

        # Cadera
        teta_c=horizontal_angle(self,6,3)

        # Rodilla
        teta_r = angle_measure(self,7,8,6)


    return [{'value':teta_b,'title':'Brazo'},{'value':teta_c,'title':'Cadera'},{'value':teta_r,'title':'Rodilla'}]

def bike_post(self):
    """
    Test for other product
    """
    # Rodilla
    teta_rd = angle_measure(self,13,12,14)

    # Rodilla
    teta_ri = angle_measure(self,7,8,6)

    return [{'value':teta_rd,'title':'Rodilla Der'},{'value':teta_ri,'title':'Rodilla Izq'}]
    

class Exercises:
    angle_calculation = {
                        'M_CL'  : cuello_lateral,
                        'M_CP'  : cuello_posterior,
                        'M_CHI' : circunduccion_hombro_izquierdo,
                        'M_CHD' : circunduccion_hombro_derecho,
                        'M_SB'  : sep_brazos,
                        'M_CCO' : circunduccion_codos,
                        'M_RC'  : rotacion_columna,
                        'M_LC'  : lateralizacion_columna,
                        'M_CCA' : circunduccion_caderas,
                        'M_CR'  : circunduccion_rodillas,

                        'E_CLI' : cuello_lateral,
                        'E_CLD' : cuello_lateral,
                        'E_CP'  : cuello_posterior,
                        'E_P'   : pecho,
                        'E_ICD' : inclinacion_columna_der,
                        'E_ICI' : inclinacion_columna_izq,
                        'E_FC'  : flexion_columna,
                        'E_GD'  : gluteo_der,
                        'E_GI'  : gluteo_izq,
                        'E_GED' : gemelos_der,
                        'E_GEI' : gemelos_izq,

                        'D_ME'  : marcha_estacionaria,
                        'D_TG'  : talon_gluteo,
                        'D_SL'  : sentadilla_lat,
                        'D_RC'  : rodilla_codo,
                        'D_BJ'  : brincos_jack,
                        'D_AA'  : paso_adelante_atras,
                        'D_B'   : boxing,
                        'D_SC'  : saltar_cuerda,

                        'A_RC'  : rotacion_cuello,
                        'A_T'   : triceps,
                        'A_A'   : aplauso,
                        'A_FC'  : flexoextension_columna,
                        'A_IF'  : inclinacion_flexion,
                        'A_BP'  : brazo_pierna,
                        'A_MT'  : mano_talon,
                        'A_FE'  : flexbrazo_extcadera,
                        'A_PS'  : parar_sentar,
                        
                        'F_SL'  : sentadilla,
                        'F_AUI' : apoyo_unipodal_izq,
                        'F_AUD' : apoyo_unipodal_der,

                        'K_SL'  : squat_lat,
                        'K_SF'  : squat_front,
                        #'K_EDF' : est_der_front,
                        #'K_EDL' : est_der_lat,
                        'K_ABP' : apoyo_brazo_pierna,
                        'K_LRD' : leg_rise_der,
                        'K_LRI' : leg_rise_izq,
                        'K_PG'  : puente_gluteo,
                        'K_PLI' : plancha_lateral_izq,
                        'K_PLD' : plancha_lateral_der,
                        'K_PU'  : push_up,
                        'K_LSFD': leg_squat_front_der,
                        'K_LSFI': leg_squat_front_izq,
                        'K_LSL' : leg_squat_lat,


                        'BF_LAT': bike_lateral,
                        'BF_POS': bike_post
                        }

    def __init__(self, tag, pose, side):
        self.tag = tag
        self.pose = pose
        self.side = side

    def calculate(self):
        if self.tag not in self.angle_calculation:
            return None
        return self.angle_calculation[self.tag](self)
