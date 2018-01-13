"""
CHEN for Pumping Test Analysis in Python




Authors:

SAUL ARCINIEGA ESPARZA
Institute of Engineering of UNAM
zaul.ae@gmail.com

JOSUE TAGO PACHECO
Earth Sciences Division, Faculty of Engineering, UNAM
josue.tago@gmail.com

ANTONIO HERNANDEZ ESPRIU
Hydrogeology Group, Earth Sciences Division, Faculty of Engineering UNAM
ahespriu@unam.mx
"""

import models
import structure
import utilities


# Set utilities to models
models.interpretation.derivative._data_validation = utilities.data_validation
models.interpretation.smooth._data_validation = utilities.data_validation

models.interpretation.derivative._solvers = utilities.linear_regression
models.interpretation.smooth._solvers = utilities.linear_regression

models.interpretation.smooth._BSKF = utilities.BSpline.BSFK

structure.wells._units = utilities.units
structure.aquifers._units = utilities.units
structure.data._units = utilities.units
