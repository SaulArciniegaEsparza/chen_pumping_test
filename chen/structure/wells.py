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
        self.pumprate = _WellData(dtype=0, name=name, description=description)
        self.pumprate.set_units('s', 'm3/s')

        # Set observation wells and piezometers
        self.wells = []

    def add_well(self):
        pass

    def convert_units(self):
        pass

    def delete_well(self):
        pass

    def delete_all_wells(self):
        pass

    def get_plot_options(self):
        pass

    def is_constant_rate(self):
        pass

    def to_dict(self):
        pass

    def update(self):
        pass

    def validate_parameters(self):
        pass

    def well_count(self):
        pass

    def wells_list(self):
        pass

    def well_id(self):
        pass

    def well_name(self):
        pass


class ObservationWell(object):
    def __init__(self, name='', description=''):
        """
        Create an observation well object

        ATTRIBUTES:
            time_units: string with the time units for this well and associated data
            len_units:  string with the length units for this well and associated data
            parameters: dictionary that contains the well parameters
                 r     [float] radial distance to pumping well in longitude units
                 d     [float] depth of well screen (from top) in length units
                 l     [float] depth of well bottom screen in length units
            drawdown:  data object that contains the drawdown data
            drawdown:  list that contains data objects with the results of applied models

        Creating a new well:
        well = ObservationWell(name='Well 1', description='First well added')
        well.drawdown.set_data(x, y, xunits='min', yunits='m')
        well.set_parameters(r=50., d=5., l=15., full=False)

        Optionally:
        well = ObservationWell(name='Well 1', description='First well added')
        well.drawdown.import_data_from_filefilename, delimiter=',', skip_header=1, xunits='min', yunits='m')
        well.set_parameters(r=50., d=5., l=15., full=False)

        Adding new data:
        well.add_data(x, derivative, dtype=2, name="ds/dt Bourdet",
                      description="First derivative with Bourdet method")
        """

        # Set general info
        self._type = 2  # observation well id
        self.time_units = 's'
        self.len_units = 'm'

        self.parameters = {'full': True,  # is full penetrating?
                           'r': 1.,  # radius, distance until pumping well in length units
                           'd': 0.,  # depth of well screen (from top) in length units
                           'l': 1.}  # depth of well bottom in length units

        # Create pumping well data
        self.drawdown = _WellData(dtype=1, name=name, description=description)
        self.drawdown.set_units(self.time_units, self.len_units)

        # Set results from models
        self.data = []

    def __getitem__(self, key):
        return(self.__dict__[key])

    def add_data(self, x, y, dtype=1, name="New data", description="New data"):
        """
        Add new data object to the actual observation well or piezometer

        INPUTS
         x            [int, float, list, tuple, ndarray] time vector
         y            [int, float, list, tuple, ndarray] data vector
         dtype        [int] type of data
                       1     drawdown
                       2     drawdown first derivative
                       3     drawdown second derivative
         name         [string] data name that is used as label for plot
         description  [string] data description that is used as ylabel for plot
        """
        assert 1 <= dtype <= 3, "Bad value for data type"
        new_data = _WellData(dtype, name=name, description=description)
        new_data.set_data(x=x, y=y, xunits=self.time_units, yunits=self.len_units)
        self.data.append(new_data)

    def convert_units(self, time_units=None, len_units=None):
        """
        Convert parameters, drawdown and data units given new units

        INPUTS
         time_units       [string] new time units. If None, actual units are used
         length_units     [string] new length units. If None, actual units are used
        """
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

    def delete_data(self, key):
        """
        Removes the data object from the associated data list
        given the data name or id
        """
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

    def get_plot_options(self):
        """
        Returns a list of plot options with the visible data to be plotted
        including observation well or piezometer drawdown
        """
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

    def data_list(self):
        """
        Returns a list with the data name
        """
        list_names = []
        for well_data in self.data:
            list_names.append(well_data.name)
        return(list_names)

    def data_id(self, name):
        """
        Returns the data id in the list of data using the data name
        Only the first data with similar names is returned
        When data name is not found, -1 is returned
        """

        idx = -1
        if type(name) is str:
            data_names = self.data_list()
            if name in data_names:
                idx = data_names.index(name)
        return(idx)

    def data_count(self):
        """
        Returns the number of data associated to the well or piezometer
        """
        return(len(self.data))

    def data_name(self, idx):
        """
        Returns data name using the data index as input
        If data idx does not exist then None is returned
        """
        name = None
        if type(idx) is int:
            n = self.data_count()
            assert 0 <= idx <= n - 1, "Bad data index"
            name = self.data[idx].name
        return(name)

    def get_data(self, key):
        """
        Returns the data object giving the data name or index
        """
        if type(key) is str:
            idx = self.data_id(key)
        elif type(key) is int:
            idx = key
        else:
            raise TypeError('key must be a string or a integer.')
        n = self.data_count()
        if 0 > idx or idx > n - 1:
            raise ValueError('Bad value for key parameter')
        return(self.data[idx])

    def reset_data(self):
        """
        Delete all the associate data
        """
        self.data = []

    def set_parameters(self, full=None, r=None, l=None, d=None, z=None):
        """
        Set well or piezometer parameters
        INPUTS:
         full   [bool] if True, well is full penetrating and depth
                 parameters are ignored in computation
         r      [float] radius to pumping well in length units
         l      [float] depth from water table to well bottom in length
                 units (only for observation wells)
         d      [float] depth from water table to well top screen in length units
                 (only for observation wells)
         z      [float] depth from water table to piezometer bottom in length units
                 (only for piezometers)
        """

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

    def to_dict(self):
        """
        Returns a list of dictionaries containing all the data in
        the well that can be used to storage the data as json format
        """
        out_dict = _deepcopy(self.__dict__)
        out_dict["drawdown"] = self.drawdown.to_dict()
        out_data = []
        for i in range(self.data_count()):
            out_data.append(self.data[i].to_dict())
        out_dict["data"] = out_data
        return(out_dict)

    def update(self, new_data):
        """
        Updates the well or piezometer object using an input dictionary
        """
        if type(new_data) is not dict:
            raise TypeError("Input parameter must be a dict")
        # Update parameters
        self._type = new_data("_type", self._type)
        self.time_units = new_data("time_units", self.time_units)
        self.len_units = new_data("len_units", self.len_units)
        self.parameters = new_data("parameters", self.parameters)
        # Update drawdown
        self.drawdown.update(new_data.get("drawdown", self.drawdown.to_dict()))
        # Update data
        if "data" in new_data:
            n = len(new_data["data"])
            if n > 1:
                self.reset_data()
                for i in range(n):
                    self.add_data(0, 0)
                    self.data[i].update(new_data["data"][i])
        # End Function

    def validate_parameters(self):
        """
        Verify well parameters and returns warnings according to
        possible errors
        OUTPUTS:
         flag       [int] if an error is detected in parameters
                      then flag is returned as 0, in other way 1
         warnings   [string] warnings text
        """

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
        """
        Create a piezometer object (works similar to Observation well)

        ATTRIBUTES:
            time_units: string with the time units for this well and associated data
            len_units:  string with the length units for this well and associated data
            parameters: dictionary that contains the well parameters
                 r     [float] radial distance to pumping well in longitude units
                 z     [float] piezometer depth in length units
            drawdown:  data object that contains the drawdown data
            drawdown:  list that contains data objects with the results of applied models

        Creating a new well:
        well = Piezometer(name='Piezometer 1', description='First piezometer added')
        well.drawdown.set_data(x, y, xunits='min', yunits='m')
        well.set_parameters(r=50., d=5., l=15., full=False)

        Optionally:
        well = Piezometer(name='Piezometer 1', description='First piezometer added')
        well.drawdown.import_data_from_file(filename, delimiter=',', skip_header=1, xunits='min', yunits='m')
        well.set_parameters(r=50., d=5., l=15., full=False)

        Adding new data:
        well.add_data(x, derivative, dtype=2, name="ds/dt Bourdet",
                      description="First derivative with Bourdet method")
                """
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


"""__________________________ WELL DATA CLASS _______________________________"""


class _WellData(object):
    def __init__(self, dtype=0, description="", name="data"):
        """
        Create well data class for storage data and results

         INPUTS:
         dtype        [int] type of data
                       0     pumping rate
                       1     drawdown
                       2     drawdown first derivative
                       3     drawdown second derivative
         name         [string] data name that is used as label for plot
         description  [string] data description that is used as ylabel for plot
        """

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

    def convert_units(self, xunits=None, yunits=None):
        """
        Conversion of data units

        INPUTS
         xunits    [string] optional new time units
         yunits    [string] optional new data units
        """
        # Call x values
        if xunits is None:
            x = self.x
            xunits = self.xunits
        else:  # convert
            x = _units.units_conversion(self.x, self.xunits, xunits)

        # Call y values
        if yunits is None:
            y = self.y
            yunits = self.yunits
        else:  # convert
            y = _units.units_conversion(self.y, self.yunits, yunits)

        # Storage converted data
        self.set_data(x, y, xunits=xunits, yunits=yunits)
        # End Function

    def get_data(self):
        """
        Get time and data arrays

        OUTPUTS
        x, y         [ndarray] time and data arrays
        """
        return(self.x.copy(), self.y.copy())

    def get_data_type(self):
        """
        Return a string with the data type (pumping rate, drawdown, derivative)
        """
        if self.dtype == 0:
            return("Pumping rate")
        elif self.dtype == 1:
            return ("Drawdown")
        else:
            return ("Drawdown derivative")

    def get_parameters(self):
        """
        Returns the model parameters as dictionary
        """
        return(_deepcopy(self._model_params))

    def get_plot_options(self):
        """
        Get attributes for plot data as a dict

        OUTPUTS:
         options     [dict] output dictionary with options for plots
        """
        options = _deepcopy(self._graph)
        options['x'] = self.x.copy()
        options['y'] = self.y.copy()
        options['label'] = self.name
        options['xlabel'] = 'Time (%s)' % (self.xunits)
        options['ylabel'] = '%s (%s)' % (self.get_data_type(), self.yunits)
        return (options)

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
        # End Function

    def set_data(self, x=None, y=None, data=None, xunits='s', yunits='m'):
        """
        Set data from x and y one-dimension arrays or form a two-dimension array

        INPUTS:
         x       [int, float, list, tuple, ndarray] one-dimension time data
         y       [int, float, list, tuple, ndarray] one-dimension pumping rate or drawdown data
         data    [list, tuple, ndarray] two-dimension data array [time, data]
         xunits  [string] time units
         yunits  [string] data units, must be consistent with data type
        NOTE: x and y must be input at the same time
        """

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

    def set_units(self, xunits=None, yunits=None):
        """
        Set units to data

        INPUTS
         xunits    [string] input time units
         yunits    [string] input data units (must be consistent with data type)
        """

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

    def set_plot_options(self, color=None, symbol=None, line=None,
                         width=None, visible=None):
        """
        Change attributes that are used to plot the data

        INPUTS
         color     [string] color string (by default black 'k')
         symbol    [string] symbol (by default circles are used 'o')
         line      [string] line style (by default no line is used '')
         width     [int, float] line width (by default 1)
         visible   [boolean] if True, data is plotted, if False is not plotted

         See matplotlib.pyplot.plot()
        """

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

    def to_dict(self):
        """
        Export data properties to a dictionary

        OUTPUTS
         data_properties     [dict] output attributes
        """
        data_properties = {'dtype': self.dtype,
                           'name': self.name,
                           'description': self.description,
                           'x': list(self.x),
                           'y': list(self.y),
                           'parameters': _deepcopy(self._model_params),
                           'plot_options': _deepcopy(self._graph)}
        return(data_properties)

    def update(self, new_data):
        """
        Update data from a dictionary

        INPUTS
         new_data    [dict] input parameters
        """
        original = self.__dict__
        fixed = dict.fromkeys(original.keys())
        for key in original.keys():
            fixed[key] = new_data.get(key, original.get(key))
        keys = ('_model_params', '_graph')
        for key in keys:
            fixed[key] = new_data.get(key, original.get(key))
        if type(fixed["x"]) is not _np.ndarray:
            fixed["x"] = _np.ndarray(fixed["x"], dtype=_np.float32)
        if type(fixed["y"]) is not _np.ndarray:
            fixed["y"] = _np.ndarray(fixed["y"], dtype=_np.float32)
        self.__dict__.update(fixed)

