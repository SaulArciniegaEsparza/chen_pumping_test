# -*- coding: utf-8 -*-
"""
CHEN pumping test analysis
Units conversion for model inputs and outputs

Functions:
    units_conversion  > Units conversion for time, drawdown and pumping rate
"""

"""_________________________ UNITS FACTORS _________________________________"""

# units data base
LEN_FACTORS = {'cm': {'cm': 1.,
                      'm':  1. / 100.,
                      'ft': 1. / 30.48,
                      'in': 1. / 2.54},
               'm': {'cm': 100.,
                     'm':  1.,
                     'ft': 3.28084,
                     'in': 39.370079},
               'ft': {'cm': 30.48,
                      'm':  1. / 3.2808,
                      'ft': 1.,
                      'in': 12.},
               'in': {'cm': 2.54,
                      'm':  1. / 39.370079,
                      'ft': 1. / 12.,
                      'in': 1.}
               }

VOL_FACTORS = {'cm3': {'cm3': 1.,
                       'm3':  1. / 100. ** 3,
                       'ft3': 1. / 30.48 ** 3,
                       'in3': 1. / 2.54 ** 3,
                       'l':   1 / 1000.},
               'm3': {'cm3': 100. ** 3,
                      'm3':  1.,
                      'ft3': 3.28084 ** 3,
                      'in3': 39.370079 ** 3,
                      'l':  1000.},
               'ft3': {'cm3': 30.48 ** 3,
                       'm3':  1. / 3.2808 ** 3,
                       'ft3': 1.,
                       'in3': 12. ** 3,
                       'l': 28.3168},
               'in3': {'cm3': 2.54 ** 3,
                       'm3':  1. / 39.370079 ** 3,
                       'ft3': 1. / 12. ** 3,
                       'in3': 1.,
                       'l':   1. / 61.0237},
               'l': {'cm3': 1000.,
                     'm3':  1. / 1000.,
                     'ft3': 1. / 28.3168,
                     'in3': 61.0237,
                     'l':   1.}
               }

TIME_FACTORS = {'s': {'s':   1.,
                      'min': 1. / 60.,
                      'hr':  1. / 3600.,
                      'day': 1 / 86400.},
                'min': {'s':   60.,
                        'min': 1.,
                        'hr':  1. / 60.,
                        'day': 1. / 1440.},
                'hr': {'s':   3600.,
                       'min': 60.,
                       'hr':  1.,
                       'day': 1. / 24.},
                'day': {'s':   86400.,
                        'min': 1440.,
                        'hr':  24.,
                        'day': 1.}
                }


# Units conversion for time and drawdown
# INPUTS
#  x          [int, float, ndarray] input values
#  in_unit    [string] input unit
#  out_unit   [string] output unit
# OUTPUTS
#  xc         [as x] output value
def units_conversion(x, in_unit='m3/day', out_unit='l/s'):
    # Check inputs
    in_unit = in_unit.lower().split('/')
    out_unit = out_unit.lower().split('/')

    n1, n2 = len(in_unit), len(out_unit)

    if n1 == 0 or n1 > 2:
        raise AssertionError('Bad argument as in_unit={}'.format(in_unit))
    if n2 == 0 or n2 > 2:
        raise AssertionError('Bad argument as out_unit={}'.format(out_unit))
    if n1 != n2:
        raise AssertionError("Units {} can't be converted to {}".format(in_unit, out_unit))

    # Check available units
    if n1 == 2:  # volumetric and time conversion
        if in_unit[0] not in VOL_FACTORS or in_unit[1] not in TIME_FACTORS:
            raise AssertionError("Bad input units {}".format(in_unit))
        if out_unit[0] not in VOL_FACTORS or out_unit[1] not in TIME_FACTORS:
            raise AssertionError("Bad output units {}".format(out_unit))
        op = 0
    else:        # length or time conversion
        if in_unit[0] in LEN_FACTORS and out_unit[0] in LEN_FACTORS:
            op = 1
        elif in_unit[0] in TIME_FACTORS and out_unit[0] in TIME_FACTORS:
            op = 2
        else:
            raise AssertionError("Units {} can't be converted to {}".format(in_unit, out_unit))

    # Convert units
    if op == 0:    # volume
        xc = x * VOL_FACTORS[in_unit[0]][out_unit[0]] / TIME_FACTORS[in_unit[1]][out_unit[1]]
    elif op == 1:  # length
        xc = x * LEN_FACTORS[in_unit[0]][out_unit[0]]
    else:          # time
        xc = x * TIME_FACTORS[in_unit[0]][out_unit[0]]

    return(xc)  # units_conversion()


# Check
def validate_units(unit):
    if type(unit) is not str:
        return(-1)

    unit = unit.lower().split('/')
    n = len(unit)

    # Check available units
    if n == 2:  # volumetric and time conversion
        if unit[0] not in VOL_FACTORS or unit[1] not in TIME_FACTORS:
            return(-1)
        return(2)
    else:  # length or time conversion
        if unit[0] in LEN_FACTORS:
            return(0)
        elif unit[0] in TIME_FACTORS:
            return(1)
        else:
            return(-1)

