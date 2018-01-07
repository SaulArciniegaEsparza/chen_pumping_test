"""
CHEN pumping test analysis
Pumping well and observation well properties
"""

import numpy as _np
from copy import deepcopy as _deepcopy


"""____________________________ WELL OBJECTS _______________________________"""


class WellPumping(object):
    def __init__(self, description='', name=''):
        self._type = 'pumpwell'
        self.parameters = {'rw': 1.,
                           'd':  0.,
                           'l':  1.}




# Pumping well default parameters
# Creates a dictionary with default parameters for a pumping well
# Parameters:
#  t     [float, ndarray] time
#  Q     [float, ndarray] pumping rate
#  rw    [float] well radius in longitude units
#  d     [float] depth of well screen (from top) as percentage between 0 and 1
#  l     [float] depth of well bottom screen as percentage between 0 and 1
# NOTES: if t and Q are floats, then a constant rate is used. If t and Q are
# ndarrays, then multiple pumping rate is used (Only few models used multiple flow)
# parameters d and l are represented as percentage of the aquifer thickness,
# then d+l=1.0, if not a error is raised
def pumpwell_parameters():
    parameters = {'type': "pumpwell",
                  't':    0.,
                  'Q':    1.,
                  'rw':   1.,
                  'd':    0.,
                  'l':    1.}
    return(parameters)  # End Function


# Observation well default parameters
# Creates a dictionary with default parameters for an observation well
# Parameters:
#  t     [ndarray] time
#  s     [ndarray] drawdown data
#  r     [float] radial distance to pumping well in longitude units
#  d     [float] depth of well screen (from top) as percentage between 0 and 1
#  l     [float] depth of well bottom screen as percentage between 0 and 1
# NOTES: parameters d and l are represented as percentage of the aquifer thickness,
# then d+l=1.0, if not a error is raised
def obswell_parameters():
    parameters = {'type': "obswell",
                  't':    _np.array([0, 1], dtype=_np.float32),
                  's':    _np.array([1, 1], dtype=_np.float32),
                  'r':    1.0,
                  'd':    0.0,
                  'l':    1.0}
    return (parameters)  # End Function


# Observation well default parameters
# Creates a dictionary with default parameters for an observation well
# Parameters:
#  t     [ndarray] time
#  s     [ndarray] drawdown data
#  r     [float] radial distance to pumping well in longitude units
#  z     [float] piezometer depth as percentage
# NOTES: z is represented as percentage of the aquifer thickness
def piezometer_parameters():
    parameters = {'type': "piezometer",
                  't':     _np.array([0, 1], dtype=_np.float32),
                  's':     _np.array([1, 1], dtype=_np.float32),
                  'r':     1.0,
                  'z':     0.0}
    return (parameters)  # End Function


def check_well_parameters(data):
    well_type = data.get('type', None)
    if well_type == "pumping well":
        pass
    elif well_type == "observation well":
        pass
    elif well_type == "observation well":
        pass
    else:
        raise TypeError("Input parameters don't correspond to a well parameters!")


"""__________________________ WELL DATA CLASS _______________________________"""


# Create well data class
class _WellData(object):
    def __init__(self, dtype="drawdown", description="", name="data"):
        # Data parameters
        self.dtype = dtype              # data type
        self.name = name                # data name
        self.description = description  # data description

        # Set data
        self.x = _np.array([], dtype=_np.float32)
        self.y = _np.array([], dtype=_np.float32)
        self.xunits = 's'
        self.yunits = 'm'

        # Set model params
        self._model_params = {}

        # Plot parameters
        self._graph = {'color':   'k',
                       'symbol':  'o',
                       'line':    '',
                       'width':   1.0,
                       'visible': True}

    # self[key] function
    def __getitem__(self, key):
        obj = self.__dict__
        if key == 'parameters':
            key = '_model_params'
        elif key == 'plot_options':
            key = '_graph'
        return(obj.get(key, None))

    # Export data properties to a dictionary
    # OUTPUT
    #  data_properties     [dict] output attributes
    def to_dict(self):
        data_properties = {'dtype': self.dtype,
                           'name': self.name,
                           'description': self.description,
                           'x': self.x.copy(),
                           'y': self.y.copy(),
                           'parameters': _deepcopy(self._model_params),
                           'plot_options': _deepcopy(self._graph)}
        return(data_properties)

    # Update data from a dictionary
    # INPUTS
    #  new_data    [dict] input parameters
    def update_data(self, new_data):
        original = self.__dict__
        fixed = dict.fromkeys(original.keys())
        for key in original.keys():
            fixed[key] = new_data.get(key, original.get(key))
        keys = ('_model_params', '_graph')
        for key in keys:
            fixed[key] = new_data.get(key, original.get(key))
        self.__dict__.update(fixed)

    # Set data from x and y one-dimension arrays or form a two-dimension array
    # INPUTS:
    #  x      [int, float, list, tuple, ndarray] one-dimension time data
    #  y      [int, float, list, tuple, ndarray] one-dimension pumping rate or drawdown data
    #  data   [list, tuple, ndarray] two-dimension data array [time, data]

    # NOTE: x and y must be input at the same time
    def set_data(self, x=None, y=None, data=None, xunits='s', yunits='m'):
        # Check inputs
        if x is not None and y is not None:
            # Convert x to a float32 data type numpy array
            if type(x) in (list, tuple):
                x = _np.array(x, _np.float32)
            elif type(x) is _np.ndarray:
                if x.dtype != _np.float32:
                    x = x.astype(_np.float32)
            elif type(x) in (int, float):
                x = _np.array([x], _np.float32)
            else:
                raise TypeError('Bad x type <{}>'.format(str(type(x))))
            if x.ndim != 1:
                raise TypeError('x must be a one-dimensional array')

            # Convert y to a float32 data type numpy array
            if type(y) in (list, tuple):
                y = _np.array(y, _np.float32)
            elif type(y) is _np.ndarray:
                if y.dtype != _np.float32:
                    y = y.astype(_np.float32)
            elif type(y) in (int, float):
                y = _np.array([y], _np.float32)
            else:
                raise TypeError('Bad y type <{}>'.format(str(type(y))))
            if y.ndim != 1:
                raise TypeError('y must be a one-dimensional array')

        if data is not None:
            # Convert data to a float32 data type numpy array
            if type(y) in (list, tuple):
                data = _np.array(data, _np.float32)
            elif type(data) is _np.ndarray:
                if data.dtype != _np.float32:
                    data = data.astype(_np.float32)
            else:
                raise TypeError('Bad data type <{}>'.format(str(type(data))))
            if data.ndim != 2:
                raise TypeError('data must be a two-dimensional array')
            x, y = data[:, 0], data[:, 1]

        # Check x and y size
        if x.size != y.size:
            return AssertionError('x and y must have the same size <{},{}>'.
                                  format(x.size, y.size))

        # Save data
        self.x, self.y = x, y
        self.xunits, self.yunits = xunits, yunits
        # End Function

    # Get time and data
    def get_data(self):
        return(self.x, self.y)

    # Get attributes for plot data
    def get_plot_options(self):
        options = _deepcopy(self._graph)
        options['label'] = self.name
        options['xlabel'] = 'Time (%s)' % (self.xunits)
        options['ylabel'] = '%s (%s)' % (self.description, self.yunits)
        return(options)

