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

    def convert_units(self):
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

        self.parameters = {'full': True,  # is full penetrating?
                           'r': 1.,  # radius, distance until pumping well in length units
                           'd': 0.,  # depth of well screen (from top) in length units
                           'l': 1.}  # depth of well bottom in length units

        # Create pumping well data
        self.drawdown = _WellData(dtype=0, name=name, description=description)
        self.drawdown.set_units(self.time_units, self.len_units)

        # Set results from models
        self.data = []

    def __getitem__(self, key):
        return(self.__dict__[key])

    # Add new data object to the actual observation well or piezometer
    # INPUTS
    #  x            [int, float, list, tuple, ndarray] time vector
    #  y            [int, float, list, tuple, ndarray] data vector
    #  dtype        [int] type of data
    #                1     drawdown
    #                2     drawdown first derivative
    #                3     drawdown second derivative
    #  name         [string] data name that is used as label for plot
    #  description  [string] data description that is used as ylabel for plot
    def add_data(self, x, y, dtype=1, name="New data", description="New data"):
        assert 1 <= dtype <= 3, "Bad value for data type"
        new_data = _WellData(dtype, name=name, description=description)
        new_data.set_data(x=x, y=y, xunits=self.time_units, yunits=self.len_units)
        self.data.append(new_data)

    # Convert parameters, drawdown and data units given new units
    # INPUTS
    #  time_units       [string] new time units. If None, actual units are used
    #  length_units     [string] new length units. If None, actual units are used
    def convert_units(self, time_units=None, len_units=None):
        in_time = self.time_units
        # Check new time units
        if time_units is None:
            time_units = in_time
        flag = _units.validate_units(time_units)
        if flag == -1:
            raise ValueError('Bad time units input {}'.format(time_units))
        # Check new length units
        in_len = self.len_units
        if len_units is None:
            len_units = in_len
        flag = _units.validate_units(len_units)
        if flag == -1:
            raise ValueError('Bad length units input {}'.format(len_units))
        # Convert parameters units
        for key, value in self.parameters.items():
            if type(value) in [int, float]:
                self.parameters[key] = _units.units_conversion(value, in_len, len_units)
        # Convert drawdown data
        self.drawdown.convert_units(time_units, len_units)
        # Convert associate data units
        for i in range(self.data_count()):
            if self.data[i].dtype == 1:    # drawdown units
                data_units = len_units
            elif self.data[i].dtype == 2:  # first derivative units
                data_units = len_units + "/" + time_units
            elif self.data[i].dtype == 3:  # second derivative units
                data_units = len_units + "/" + time_units + "2"
            self.data[i].convert_units(time_units, data_units)
        # End Function

    # Removes the data object from the associated data list
    # given the data name or id
    def delete_data(self, key):
        if type(key) is str:
            idx = self.data_id(key)
        elif type(key) is int:
            idx = key
        else:
            raise TypeError('key must be a string or a integer.')
        n = self.data_count()
        if 0 <= idx <= n - 1:
            raise ValueError('Bad value for key parameter')
        del(self.data[idx])

    # Returns a list of plot options with the visible data to be plotted
    # including observation well or piezometer drawdown
    def get_plot_options(self):
        plot_options = []
        # Get drawdown plot options
        op = self.drawdown.get_plot_options()
        if op['visible']:
            plot_options.append(op)
        # Get associated data options
        for i in range(self.data_count()):
            op = self.data[i].get_plot_options()
            if op['visible']:
                plot_options.append(op)
        return(plot_options)

    # Returns a list with the data name
    def data_list(self):
        list_names = []
        for well_data in self.data:
            list_names.append(well_data.name)
        return(list_names)

    # Returns the data id in the list of data using the data name
    # Only the first data with similar names is returned
    # When data name is not found, -1 is returned
    def data_id(self, name):
        idx = -1
        if type(name) is str:
            data_names = self.data_list()
            if name in data_names:
                idx = data_names.index(name)
        return(idx)

    # Returns the number of data associated to the well or piezometer
    def data_count(self):
        return(len(self.data))

    # Returns data name using the data index as input
    # If data idx does not exist then None is returned
    def data_name(self, idx):
        name = None
        if type(idx) is int:
            n = self.data_count()
            assert 0 <= idx <= n - 1, "Bad data index"
            name = self.data[idx].name
        return(name)

    # Returns the data object giving the data name or index
    def get_data(self, key):
        if type(key) is str:
            idx = self.data_id(key)
        elif type(key) is int:
            idx = key
        else:
            raise TypeError('key must be a string or a integer.')
        n = self.data_count()
        if 0 <= idx <= n - 1:
            raise ValueError('Bad value for key parameter')
        return(self.data[idx])

    # Delete all the associate data
    def reset_data(self):
        self.data = []

    # Set well parameters
    # INPUTS:
    #  full   [bool] if True, well is full penetrating and depth
    #          parameters are ignored in computation
    #  r      [float] radius to pumping well in length units
    #  l      [float] depth from water table to well bottom in length
    #          units (only for observation wells)
    #  d      [float] depth from water table to well top screen in length units
    #          (only for observation wells)
    #  z      [float] depth from water table to piezometer bottom in length units
    #          (only for piezometers)
    def set_parameters(self, full=None, r=None, l=None, d=None, z=None):
        original = _deepcopy(self.parameters)  # save in case of error

        if type(full) is bool:
            self.parameters["full"] = full
        if type(r) in [int, float]:
            self.parameters["r"] = float(r)
        if self._type == 2:  # observation well
            if type(d) in [int, float]:
                self.parameters["d"] = float(d)
            if type(l) in [int, float]:
                self.parameters["l"] = float(l)
        else:                # piezometer
            if type(z) in [int, float]:
                self.parameters["z"] = float(z)

        flag, message = self.validate_parameters()
        if flag == 0:
            print(message)
            self.parameters.update(original)
        # End Function

    # Returns a list of dictionaries containing all the data in
    # the well that can be used to storage the data as json format
    def to_dict(self):
        out_dict = self.__dict__
        out_dict["drawdown"] = self.drawdown.to_dict()
        out_data = []
        for i in range(self.data_count()):
            out_data.append(self.data[i].to_dict())
        out_dict["data"] = out_data
        return(out_dict)

    # Updates the well or piezometer object using an input dictionary
    def update(self, new_data):
        if type(new_data) is not dict:
            raise TypeError("Input parameter must be a dict")
        # Update parameters
        self._type = new_data("_type", self._type)
        self.time_units = new_data("time_units", self.time_units)
        self.len_units = new_data("len_units", self.len_units)
        self.parameters = new_data("parameters", self.parameters)
        # Update drawdown
        self.drawdown = new_data.get("drawdown", self.drawdown.to_dict())
        # Update data
        if "data" in new_data:
            n = len(new_data["data"])
            if n > 1:
                self.reset_data()
                for i in range(n):
                    self.add_data(0, 0)
                    self.data[i].update(new_data["data"][i])
        # End Function

    # Verify well parameters and returns warnings according to
    # possible errors
    # OUTPUTS:
    #  flag       [int] if an error is detected in parameters
    #               then flag is returned as 0, in other way 1
    #  warnings   [string] warnings text
    def validate_parameters(self):
        flag = 1
        warnings = ""
        # Check radius
        r = self.parameters.get('r', 0)
        if type(r) not in [int, float]:
            flag = 0
            warnings += "Radius r must be a float value\n"
        else:
            if r <= 0:
                flag = 0
                warnings += "Radius r must be higher than 0\n"
        # Check if is full penetrating
        op = self.parameters.get('full', False)

        if ~op:
            # Check observation well length
            if 'd' in self.parameters and 'l' in self.parameters:
                d = self.parameters.get('d', -1)
                l = self.parameters.get('l', -1)
                if type(l) not in [int, float]:
                    flag = 0
                    warnings += "Depth of well bottom must be a float value\n"
                else:
                    if l < 0:
                        flag = 0
                        warnings += "Depth l must be higher than 0\n"
                if type(d) not in [int, float]:
                    flag = 0
                    warnings += "Depth of well screen must be a float value\n"
                else:
                    if d < 0 or d > l:
                        flag = 0
                        warnings += "Depth d must be in range 0 <= d <= l\n"
            # Check piezometer depth
            elif 'z' in self.parameters:
                z = self.parameters.get('z', -1)
                if type(z) not in [int, float]:
                    flag = 0
                    warnings += "Depth of piezometer must be a float value\n"
                else:
                    if z < 0:
                        flag = 0
                        warnings += "Depth z must be higher than 0\n"
            else:
                flag = 0
                warnings += "Well don't contain well depth attributes\n"
        return(flag, warnings)  # End Function


class Piezometer(ObservationWell):
    def __init__(self, description='', name=''):
        super(Piezometer, self).__init__()

        # Set general info
        self._type = 3  # piezometer id
        self.parameters = {'full': True,  # is full penetrating?
                           'r': 1.,  # distance until pumping well in length units
                           'z': 1.}  # piezometer depth in length units

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
#  description  [string] data description that is used as ylabel for plot
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
    def convert_units(self, xunits=None, yunits=None):
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

    def import_data_from_file(self, filename, delimiter=',', skip_header=1,
                              xunits='s', yunits='m'):
        """
        Load time and data from a delimited file

        INPUTS:
          filename           [string] delimited file. Must contain two columns. First
                              column is loaded as time and second column as data
          delimiter          [string] text delimiter. By default ','
          skip_header        [int] number of lines to ignore at the file header
          xunits             [string] time units
          yunits             [string] data units
        """
        data = _np.genfromtxt(filename, dtype=_np.float32,
                              delimiter=delimiter, skip_header=skip_header)
        self.set_data(data=data, xunits=xunits, yunits=yunits)

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
            raise TypeError('Error type drawdown first derivative units {}'.
                            format(yunits))
        elif self.dtype == 3 and flag_yunits != 4:  # drawdown derivative data
            raise TypeError('Error type drawdown second derivative units {}'.
                            format(yunits))
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
    def set_plot_options(self, color=None, symbol=None, line=None,
        width=None, visible=None):
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

    # Get model parameters
    # OUTPUTS
    #  params      [dict] model parameters
    def get_parameters(self):
        return(self._model_params)

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
    def update(self, new_data):
        original = self.__dict__
        fixed = dict.fromkeys(original.keys())
        for key in original.keys():
            fixed[key] = new_data.get(key, original.get(key))
        keys = ('_model_params', '_graph')
        for key in keys:
            fixed[key] = new_data.get(key, original.get(key))
        self.__dict__.update(fixed)

