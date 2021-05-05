import modules.ex_score.estiramientos.E_CLI     as e_cli    #Cuello izquierda
import modules.ex_score.estiramientos.E_CLD     as e_cld    #Cuello derecha
import modules.ex_score.estiramientos.E_CP      as e_cp     #Cuello posterior
import modules.ex_score.estiramientos.E_P       as e_p      #Pecho
import modules.ex_score.estiramientos.E_IC      as e_ic     #Inclinacion columna
import modules.ex_score.estiramientos.E_FC      as e_fc     #Flexion columna
import modules.ex_score.estiramientos.E_G       as e_g      #Gluteos
import modules.ex_score.estiramientos.E_GE      as e_ge     #Gemelos

import modules.ex_score.movilidad.M_CL          as m_cl     #Cuello lateral
import modules.ex_score.movilidad.M_CP          as m_cp     #Cuello posterior
import modules.ex_score.movilidad.M_CH          as m_ch     #Circunduccion hombros
import modules.ex_score.movilidad.M_SB          as m_sb     #Separacion brazos
import modules.ex_score.movilidad.M_CCO         as m_cco    #Circunduccion codos
import modules.ex_score.movilidad.M_LC          as m_lc     #Lateralizacion columna
import modules.ex_score.movilidad.M_CCA         as m_cca    #Circunduccion caderas
import modules.ex_score.movilidad.M_CR          as m_cr     #Circunduccion rodillas
import modules.ex_score.movilidad.M_RC          as m_rc     #Rotacion de columna

import modules.ex_score.dinamicos.D_ME          as d_me     #Marcha estacionaria
import modules.ex_score.dinamicos.D_TG          as d_tg     #Talon a gluteo
import modules.ex_score.dinamicos.D_SL          as d_sl     #Sentadilla lateral
import modules.ex_score.dinamicos.D_BJ          as d_bj     #Brincos jack
import modules.ex_score.dinamicos.D_AA          as d_aa     #Paso adelante atras
import modules.ex_score.dinamicos.D_B           as d_b      #Boxing
import modules.ex_score.dinamicos.D_SC          as d_sc     #Saltar cuerda
import modules.ex_score.dinamicos.D_RC          as d_rc     #Rodilla codo

import modules.ex_score.adultos.A_A             as a_a      #Aplauso
import modules.ex_score.adultos.A_BP            as a_bp     #Brazo Pierna
import modules.ex_score.adultos.A_FC            as a_fc     #Flexoextension de columna
import modules.ex_score.adultos.A_FE            as a_fe     #Flexbrazo extcadera
import modules.ex_score.adultos.A_IF            as a_if     #Inclinacion Flexion
import modules.ex_score.adultos.A_RC            as a_rc     #Rotacion cuello
import modules.ex_score.adultos.A_PS            as a_ps     #Parar sentar
import modules.ex_score.adultos.A_T             as a_t      #Triceps
import modules.ex_score.adultos.A_MT            as a_mt     #Mano-Talon 

import modules.ex_score.funcionales.F_AU            as f_au     #Apoyo unipodal
import modules.ex_score.funcionales.F_SL            as f_sl     #Sentadilla (funcional)

def cuello_lateral_izquierdo(self):

    data = self.data
    result = e_cli.score(data)
    return result

def cuello_lateral_derecho(self):

    data = self.data
    result = e_cld.score(data)
    return result

def cuello_posterior(self):

    data = self.data
    result = e_cp.score(data)
    return result

def pecho(self):

    data = self.data
    result = e_p.score(data)
    return result

def inclinacion_columna(self):

    data = self.data
    result = e_ic.score(data)
    return result

def flexion_columna(self):

    data = self.data
    result = e_fc.score(data)
    return result

def gluteo(self):

    data = self.data
    result = e_g.score(data)
    return result

def gemelos(self):

    data = self.data
    result = e_ge.score(data)
    return result

def cuello_lateral(self):

    data = self.data
    result = m_cl.score(data)
    return result

def cuello_posterior_movilidad(self):

    data = self.data
    result = m_cp.score(data)
    return result

def circunduccion_hombro(self):

    data = self.data
    result = m_ch.score(data)
    return result


def sep_brazos(self):

    data = self.data
    result = m_sb.score(data)
    return result

def circunduccion_codos(self):

    data = self.data
    result = m_cco.score(data)
    return result


def lateralizacion_columna(self):

    data = self.data
    result = m_lc.score(data)
    return result


def circunduccion_caderas(self):

    data = self.data
    result = m_cca.score(data)
    return result

def circunduccion_rodillas(self):

    data = self.data
    result = m_cr.score(data)
    return result

def marcha_estacionaria(self):
    
    data = self.data
    result = d_me.score(data)
    return result

def talon_gluteo(self):
    
    data = self.data
    result = d_tg.score(data)
    return result

def parar_sentar(self):
    
    data = self.data
    result = a_ps.score(data)
    return result

def sentadilla_lat(self):
    
    data = self.data
    result = d_sl.score(data)
    return result

def brincos_jack(self):
    
    data = self.data
    result = d_bj.score(data)
    return result

def paso_adelante_atras(self):
    
    data = self.data
    result = d_aa.score(data)
    return result

def boxing(self):
    
    data = self.data
    result = d_b.score(data)
    return result

def saltar_cuerda(self):
    
    data = self.data
    result = d_sc.score(data)
    return result

def rodilla_codo(self):
    
    data = self.data
    result = d_rc.score(data)
    return result

def rotacion_cuello(self):
    
    data = self.data
    result = a_rc.score(data)
    return result

def triceps(self):
    
    data = self.data
    result = a_t.score(data)
    return result

def aplauso(self):
    
    data = self.data
    result = a_a.score(data)
    return result

def flexoextension_columna(self):
    
    data = self.data
    result = a_fc.score(data)
    return result

def inclinacion_flexion(self):
    
    data = self.data
    result = a_if.score(data)
    return result

def brazo_pierna(self):
    
    data = self.data
    result = a_bp.score(data)
    return result

def mano_talon(self):
    
    data = self.data
    result = a_mt.score(data)
    return result

def flexbrazo_extcadera(self):
    
    data = self.data
    result = a_fe.score(data)
    return result

def apoyo_unipodal(self):
    data = self.data
    result = f_au.score(data)
    return result

def sentadilla(self):
    data = self.data
    result = f_sl.score(data)
    return result

def rotacion_columna(self):
    data = self.data
    result = m_rc.score(data)
    return result

class Exercises:
    exercise_scoring = {
    
                        'E_CLI'  : cuello_lateral_izquierdo,
                        'E_CLD'  : cuello_lateral_derecho,
                        'E_CP'  : cuello_posterior,
                        'E_P'   : pecho,
                        'E_ICD' : inclinacion_columna,
                        'E_ICI' : inclinacion_columna,
                        'E_FC'  : flexion_columna,
                        'E_GD'  : gluteo,
                        'E_GI'  : gluteo,
                        'E_GED' : gemelos,
                        'E_GEI' : gemelos,

                        'M_CL'  : cuello_lateral,
                        'M_CP'  : cuello_posterior_movilidad,
                        'M_CHI' : circunduccion_hombro,
                        'M_CHD' : circunduccion_hombro,
                        'M_SB'  : sep_brazos,
                        'M_CCO' : circunduccion_codos,
                        'M_LC'  : lateralizacion_columna,
                        'M_CCA' : circunduccion_caderas,
                        'M_CR'  : circunduccion_rodillas,
                        'M_RC'  : rotacion_columna,

                        'D_ME'  : marcha_estacionaria,
                        'D_TG'  : talon_gluteo,
                        'D_SL'  : sentadilla_lat,
                        'D_BJ'  : brincos_jack,
                        'D_AA'  : paso_adelante_atras,
                        'D_B'   : boxing,
                        'D_SC'  : saltar_cuerda,
                        'D_RC'  : rodilla_codo,

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
                        'F_AUI' : apoyo_unipodal,
                        'F_AUD' : apoyo_unipodal
                         }

    def __init__(self, tag, data):
        self.tag = tag
        self.data = data

    def calculate(self):
        if self.tag not in self.exercise_scoring:
            return None
        return self.exercise_scoring[self.tag](self)
