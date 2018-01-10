# -*- coding: utf-8 -*-
"""
CHEN pumping test analysis
Units conversion for model inputs and outputs

Constants:
    LEN_FACTORS
    VOL_FACTORS
    TIME_FACTORS

Functions:
    units_conversion  > Units conversion for time, drawdown and pumping rate
    validate_units    > Verify if an input unit is valid and return a code according to the unit type
"""

"""_________________________ UNITS FACTORS _________________________________"""


# Length factors LEN_FACTORS[in_units][out_units]
LEN_FACTORS = {'cm': {'cm': 1.,
                      'm': 1. / 100.,
                      'ft': 1. / 30.48,
                      'in': 1. / 2.54},
               'm': {'cm': 100.,
                     'm': 1.,
                     'ft': 3.28084,
                     'in': 39.370079},
               'ft': {'cm': 30.48,
                      'm': 1. / 3.2808,
                      'ft': 1.,
                      'in': 12.},
               'in': {'cm': 2.54,
                      'm': 1. / 39.370079,
                      'ft': 1. / 12.,
                      'in': 1.}
               }

# Volumetric factors VOL_FACTORS[in_units][out_units]
VOL_FACTORS = {'cm3': {'cm3': 1.,
                       'm3': 1. / 100. ** 3,
                       'ft3': 1. / 30.48 ** 3,
                       'in3': 1. / 2.54 ** 3,
                       'l': 1 / 1000.},
               'm3': {'cm3': 100. ** 3,
                      'm3': 1.,
                      'ft3': 3.28084 ** 3,
                      'in3': 39.370079 ** 3,
                      'l': 1000.},
               'ft3': {'cm3': 30.48 ** 3,
                       'm3': 1. / 3.2808 ** 3,
                       'ft3': 1.,
                       'in3': 12. ** 3,
                       'l': 28.3168},
               'in3': {'cm3': 2.54 ** 3,
                       'm3': 1. / 39.370079 ** 3,
                       'ft3': 1. / 12. ** 3,
                       'in3': 1.,
                       'l': 1. / 61.0237},
               'l': {'cm3': 1000.,
                     'm3': 1. / 1000.,
                     'ft3': 1. / 28.3168,
                     'in3': 61.0237,
                     'l': 1.}
               }

TIME_FACTORS = {'s': {'s': 1.,
                      'min': 1. / 60.,
                      'hr': 1. / 3600.,
                      'day': 1 / 86400.},
                'min': {'s': 60.,
                        'min': 1.,
                        'hr': 1. / 60.,
                        'day': 1. / 1440.},
                'hr': {'s': 3600.,
                       'min': 60.,
                       'hr': 1.,
                       'day': 1. / 24.},
                'day': {'s': 86400.,
                        'min': 1440.,
                        'hr': 24.,
                        'day': 1.}
                }


"""_________________________ UNITS METHODS _________________________________"""


def units_conversion(x, in_unit='m3/day', out_unit='l/s'):
    """
    Units conversion for time, drawdown, drawdown derivative, and pumping rate

    INPUTS:
     x          [int, float, ndarray] input values
     in_unit    [string] input unit
     out_unit   [string] output unit
    OUTPUTS:
     xc         [similar as x] output value
    """

    # Check inputs
    flag_in = validate_units(in_unit)
    flag_out = validate_units(out_unit)

    if flag_in == -1:
        raise AssertionError('Bad argument as in_unit={}'.format(in_unit))
    if flag_out == -1:
        raise AssertionError('Bad argument as out_unit={}'.format(out_unit))
    if flag_in != flag_out:
        raise AssertionError("Units {} can't be converted to {}".
                             format(in_unit, out_unit))
    # Convert units
    in_unit = in_unit.lower().split('/')
    out_unit = out_unit.lower().split('/')
    if flag_in == 0:    # length
        xc = x * LEN_FACTORS[in_unit[0]][out_unit[0]]
    elif flag_in == 1:  # time
        xc = x * TIME_FACTORS[in_unit[0]][out_unit[0]]
    elif flag_in == 2:  # pumping rate
        xc = (x * VOL_FACTORS[in_unit[0]][out_unit[0]] /
              TIME_FACTORS[in_unit[1]][out_unit[1]])
    elif flag_in == 3:  # first derivative
        xc = (x * LEN_FACTORS[in_unit[0]][out_unit[0]] /
              TIME_FACTORS[in_unit[1]][out_unit[1]])
    elif flag_in == 4:  # second derivative
        xc = (x * VOL_FACTORS[in_unit[0]][out_unit[0]] /
              (TIME_FACTORS[in_unit[1]][out_unit[1]] ** 2.0))
    return(xc)  # units_conversion()


def validate_units(unit):
    """
    Verify if an input unit is valid and return a code according to the unit type
    INPUTS
     unit     [string] input unit
    OUTPUT
     flag     [int] output flag
               -1  is not a valid unit
                0  length unit [length]
                1  time unit [time]
                2  pumping rate unit [length^3/time]
                3  first derivative unit [length/time]
                4  second derivative unit [length/time]
    """

    if type(unit) is not str:
        return(-1)

    unit = unit.lower().split('/')
    n = len(unit)

    # Check available units
    if n == 2:
        if unit[0] in VOL_FACTORS and unit[1] in TIME_FACTORS:
            return(2)  # pumping rate units
        elif unit[0] in LEN_FACTORS and unit[1] in TIME_FACTORS:
            return(3)  # first derivate units
        elif unit[0] in LEN_FACTORS and unit[1] in TIME_FACTORS + '2':
            return(4)  # second derivate units
        else:
            return(-1)
    elif n == 1:
        if unit[0] in LEN_FACTORS:
            return(0)  # length units
        elif unit[0] in TIME_FACTORS:
            return(1)  # time units
        else:
            return(-1)
    else:
        return(-1)
    # End Function

