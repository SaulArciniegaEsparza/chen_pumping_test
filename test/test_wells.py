"""
Test well structure methods to find errors
"""

import os
from os.path import dirname as up
import sys

# This file path
PATH1 = up(__file__)
# Data path
PATH2 = os.path.join(PATH1, "data_example")
# Module path
PATH3 = os.path.join(up(PATH1), r"chen\structure")


# Importar modulo
sys.path.insert(0, PATH3)
from wells import *
import numpy as np


# Create a Pumping well
NAME = 'well 1'
DESC = 'Pozo de bombeo 1'
U = ['min', 'm', 'm3/s']  # unidades
Q = 10         # bombeo (m3/s)
PARAMS = dict(full=True,    # totalmente penetrante
              rw=0.3)     # radio del pozo (m)

PW = PumpingWell(name=NAME, description=DESC, time_units=U[0], len_units=U[1],
                 pump_units=U[2])
PW.pumprate.set_data(0, 10)
PW.set_parameters(**PARAMS)


# Agregar nuevo pozo de observacion
NAME = 'P 311'
DESC = 'Pozo de observacion'
WTYPE = 0  # definir pozo de observacion
PARAMS = dict(full=True,   # totalmente penetrante
              r=50)        # distancia hasta el pozo de bombeo
FILE1 = os.path.join(PATH2, 'data1.csv')  # archivo de datos

PW.add_well(wtype=WTYPE, name=NAME, description=DESC)
OW = PW.wells[-1]  # obtener ultimo pozo, recien creado
OW.set_parameters(**PARAMS)
OW.drawdown.import_data_from_file(FILE1, delimiter=',')

x, y = OW.drawdown.get_data()
print(x, y)  # imprimir datos


#  Agregar nuevo resultado
DTYPE = 1
NAME = 'Drawdown'
DESC = 'Copia de abatimiento'
PARAMS = dict(model='Derivada de Bourdet',
              k='k')

OW.add_data(x, y, dtype=DTYPE, name=NAME, description=DESC)
OW_data = OW.data[-1]  # obtener ultimos datos
OW_data.set_parameters(PARAMS)

