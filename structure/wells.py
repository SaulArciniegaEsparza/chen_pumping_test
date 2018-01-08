"""
CHEN pumping test analysis
Pumping well and observation well properties
"""

import numpy as _np
from copy import deepcopy as _deepcopy
import units as _units


"""____________________________ WELL OBJECTS _______________________________"""


class PumpingWell(object):
    def __init__(self, description='', name=''):
        # Set general info
        self._type = 1  # pumping well id
        self.parameters = {'rw': 1.,
                           'd':  0.,
                           'l':  1.}

        # Create pumping well data
        self.data = _WellData(dtype=0, name=name, description=description)
        self.data.set_units('s', 'm3/s')

        # Set observation wells and piezometers
        self.wells = []

    def validate_parameters(self):
        pass

    def add_well(self):
        pass

    def convert_data_units(self):
        pass

    def delete_well(self):
        pass

    def is_constant_rate(self):
        pass

    def to_dict(self):
        pass

    def wells_list(self):
        pass

    def wells_id(self):
        pass

    def wells_name(self):
        pass


class ObservationWell(object):
    def __init__(self, description='', name=''):
        # Set general info
        self._type = 2  # observation well id
        self.time_units = 's'
        self.len_units = 'm'

        self.parameters = {'r': 1.,
                           'd': 0.,
                           'l': 1.}

        # Create pumping well data
        self.drawdown = _WellData(dtype=0, name=name, description=description)
        self.drawdown.set_units(self.time_units, self.len_units)

        # Set results from models
        self.results = []

    def __getitem__(self, key):
        return(self.__dict__[key])

    def add_data(self):
        pass

    def convert_parameter_units(self):
        pass

    def convert_data_units(self):
        pass

    def delete_data(self):
        pass

    def get_plot_options(self):
        pass

    def import_data(self):
        pass

    def data_list(self):
        pass

    def data_id(self):
        pass

    def data_name(self):
        pass

    def set_parameters(self):
        pass

    def to_dict(self):
        pass

    def update(self):
        pass

    def validate_parameters(self):
        pass


class Piezometer(ObservationWell):
    def __init__(self, description='', name=''):
        super(Piezometer, self).__init__()

        # Set general info
        self._type = 3  # piezometer id
        self.parameters = {'r': 1.,
                           'z': 1.}

        # Set data
        self.data.name = name
        self.data.description = description


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
#  ATTRIBUTES:
#  dtype        [int] type of data
#                0     pumping rate
#                1     drawdown
#                2     drawdown first derivative
#                3     drawdown second derivative
#  name         [string] data name that is used as label for plot
#  description  [string] data name that is used as label for plot
#  x            [ndarray] time vector
#  y            [ndarray] data vector
class _WellData(object):
    def __init__(self, dtype=0, description="", name="data"):
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

    # Update data from a dictionary
    # INPUTS
    #  new_data    [dict] input parameters
    def update(self, new_data):
        original = self.__dict__
        fixed = dict.fromkeys(original.keys())
        for key in original.keys():
            fixed[key] = new_data.get(key, original.get(key))
        keys = ('_model_params', '_graph')
        for key in keys:
            fixed[key] = new_data.get(key, original.get(key))
        self.__dict__.update(fixed)

    # Get attributes for plot data as a dict
    # OUTPUTS
    #  options     [dict] output dictionary with options for plots
    def get_plot_options(self):
        options = _deepcopy(self._graph)
        options['x'] = self.x
        options['y'] = self.y
        options['label'] = self.name
        options['xlabel'] = 'Time (%s)' % (self.xunits)
        options['ylabel'] = '%s (%s)' % (self.description, self.yunits)
        return (options)

    # Conversion of data units
    # INPUTS
    #  xunits    [string] optional new time units
    #  yunits    [string] optional new data units
    def convert_data_units(self, xunits=None, yunits=None):
        # Call x values
        if xunits is None:
            x = self.x
            xunits = self.xunits
        else:  # convert
            x = _units.units_conversion(self.x, self.xunits, xunits)

        # Call y values
        if xunits is None:
            y = self.y
            yunits = self.yunits
        else:  # convert
            y = _units.units_conversion(self.y, self.xunits, xunits)

        # Storage converted data
        self.set_data(x, y, xunits=xunits, yunits=yunits)
        # End Function

    def import_data(self):
        pass

    # Set data from x and y one-dimension arrays or form a two-dimension array
    # INPUTS:
    #  x       [int, float, list, tuple, ndarray] one-dimension time data
    #  y       [int, float, list, tuple, ndarray] one-dimension pumping rate or drawdown data
    #  data    [list, tuple, ndarray] two-dimension data array [time, data]
    #  xunits  [string] time units
    #  yunits  [string] data units, must be consistent with data type
    # NOTE: x and y must be input at the same time
    def set_data(self, x=None, y=None, data=None, xunits='s', yunits='m'):
        # Set units to data
        self.set_units(xunits, yunits)

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
        # End Function

    # Set units to data
    # INPUTS
    #  xunits    [string] input time units
    #  yunits    [string] input data units (must be consistent with data type)
    def set_units(self, xunits=None, yunits=None):
        # Check time units
        flag_xunits = _units.validate_units(xunits)
        if flag_xunits != 1:
            raise TypeError('Error type time units {}'.format(xunits))
        else:
            self.xunits = xunits

        # Check data units
        flag_yunits = _units.validate_units(yunits)
        if self.dtype == 0 and flag_yunits != 2:  # pumping well data
            raise TypeError('Error type pumping well units {}'.format(yunits))
        elif self.dtype == 1 and flag_yunits != 0:  # drawdown data
            raise TypeError('Error type drawdown units {}'.format(yunits))
        elif self.dtype == 2 and flag_yunits != 3:  # drawdown derivative data
            raise TypeError('Error type drawdown first derivative units {}'.format(yunits))
        elif self.dtype == 3 and flag_yunits != 4:  # drawdown derivative data
            raise TypeError('Error type drawdown second derivative units {}'.format(yunits))
        else:
            self.yunits = yunits
        # End Function

    # Change plot parameters
    # See matplotlib.pyplot.plot()
    # INPUTS
    #  color     [string] color string (by default black 'k')
    #  symbol    [string] symbol (by default circles are used 'o')
    #  line      [string] line style (by default no line is used '')
    #  width     [int, float] line width (by default 1)
    #  visible   [boolean] if True, data is plotted, if False is not plotted
    def set_plot_parameters(self, color=None, symbol=None, line=None, width=None, visible=None):
        if type(color) is str:
            self._graph['color'] = color
        if type(symbol) is str:
            self._graph['symbol'] = symbol
        if type(line) is str:
            self._graph['line'] = line
        if type(width) in [int, float]:
            self._graph['width'] = width
        if type(visible) is bool:
            self._graph['visible'] = visible

    # Get time and data
    # OUTPUTS
    # x, y         [ndarray] time and data arrays
    def get_data(self):
        return(self.x, self.y)

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

