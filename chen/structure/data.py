"""
CHEN pumping test analysis
Create a data object class

CLASSES:
    WellData          > Object used to store different data (pumping rate, drawdown, first and second derivative)

See help for more information about how to work with each object


Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City, Mexico
"""

import numpy as _np
from copy import deepcopy as _deepcopy


"""_______________________________ DATA CLASS _______________________________"""


class Data(object):
    def __init__(self, dtype=0, name="data", description="", xunits="s",
                 yunits="m"):
        """
        Create data class for storage data and results

         INPUTS:
         dtype        [int] type of data
                       0     pumping rate
                       1     drawdown vs time
                       2     drawdown first derivative vs time
                       3     drawdown second derivative vs time
                       4     drawdown vs distance
         name         [string] data name that is used as label for plot
         description  [string] data description that is used as ylabel for plot
         xunits       [string] x axis units
         yunits       [string] y axis units
        """

        # Data parameters
        self.dtype = dtype              # data type
        self.name = name                # data name
        self.description = description  # data description

        # Set data
        self.x = _np.array([], dtype=_np.float32)
        self.y = _np.array([], dtype=_np.float32)
        self.xunits = xunits
        self.yunits = yunits

        # Set model params
        self.parameters = {}

        # Plot parameters
        self._graph = {'color':   'k',
                       'symbol':  'o',
                       'line':    '',
                       'width':   1.0,
                       'visible': True}

    # self[key] function
    def __getitem__(self, key):
        obj = self.__dict__
        if key == 'plot_options':
            key = '_graph'
        return(obj.get(key, None))

    def __repr__(self):
        return("Data(dtype={}, name='{}')".format(self.dtype, self.name))

    def __str__(self):
        text = "________________Data Class_______________\n\n"
        text += "  Data type: {}  ({})\n".format(self.dtype, self.get_data_type())
        text += "  Well name: {}\n".format(self.name)
        text += "Description: {}\n".format(self.description)
        text += "    X Units: {}\n".format(self.xunits)
        text += "    Y Units: {}\n".format(self.yunits)
        text += "_________________________Data attributes\n"
        text += "      Length: {}\n".format(len(self.x))
        text += "   X minimum: {}\n".format(self.x[0])
        text += "   X maximum: {}\n".format(self.x[-1])
        text += "  Data range: {} - {}\n".format(max(self.y), min(self.y))
        text += "______________________________Parameters\n"
        for key, value in self.parameters.items():
            text += "  {}: {}\n".format(key, value)
        return(text)

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

    def filter_xvalues(self, start=None, end=None):
        """
        Extracts the data contained between a range of x values.

        INPUTS:
          start      [float] initial x value in the same units that data.
                      If None, start parameter is ignored.
          end        [end] final x value in the same units that data
                      If None, end parameter is ignored.
        """
        if start is None:
            start = self.x[0]
        if end is None:
            end = self.x[-1]
        pos = _np.where((start <= self.x) & (self.x <= end))[0]
        if len(pos) > 0:
            self.set_data(self.x[pos], self.y[pos])

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
            return("Drawdown vs time")
        elif self.dtype == 2:
            return("First drawdown derivative")
        elif self.dtype == 3:
            return("First drawdown derivative")
        elif self.dtype == 4:
            return("Drawdown vs distance")

    def get_parameters(self):
        """
        Returns the model parameters as dictionary
        """
        return(_deepcopy(self.parameters))

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
        options['dtype'] = self.dtype
        return(options)

    def from_file(self, filename, delimiter=',', skip_header=1,
                  xunits=None, yunits=None):
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
        if xunits is None:
            xunits = self.xunits
        if yunits is None:
            yunits = self.yunits
        data = _np.genfromtxt(filename, dtype=_np.float32,
                              delimiter=delimiter, skip_header=skip_header)
        self.set_data(data=data, xunits=xunits, yunits=yunits)
        # End Function

    def set_data(self, x=None, y=None, data=None, xunits=None, yunits=None):
        """
        Set data from x and y one-dimension arrays or form a two-dimension array

        INPUTS:
         x       [int, float, list, tuple, ndarray] one-dimension x data
         y       [int, float, list, tuple, ndarray] one-dimension y data
         data    [list, tuple, ndarray] two-dimension data array [x, y]
         xunits  [string] x units. If None, original units are used
         yunits  [string] data units, must be consistent with data type.
                  If None, original units are used
        NOTE: x and y must be input at the same time
        """

        # Set units to data
        if xunits is None:
            xunits = self.xunits
        if yunits is None:
            yunits = self.yunits
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
            raise AssertionError('x and y must have the same size <{},{}>'.
                                  format(x.size, y.size))
        # Save data
        self.x, self.y = x, y
        # End Function

    def set_units(self, xunits=None, yunits=None):
        """
        Set units to data

        INPUTS
         xunits    [string] input x axis units
         yunits    [string] input data units (must be consistent with data type)
        """

        # Check time units
        flag_xunits = _units.validate_units(xunits)
        if self.dtype <= 3 and flag_xunits != 1:
            raise TypeError('Error in time units {}'.format(xunits))
        elif self.dtype == 4 and flag_xunits != 0:
            raise TypeError('Error in distance units {}'.format(xunits))
        else:
            self.xunits = xunits

        # Check data units
        flag_yunits = _units.validate_units(yunits)
        if self.dtype == 0 and flag_yunits != 2:  # pumping data
            raise TypeError('Error in pumping well units {}'.format(yunits))
        elif (self.dtype == 1 or self.dtype == 4) and flag_yunits != 0:  # drawdown data
            raise TypeError('Error in drawdown units {}'.format(yunits))
        elif self.dtype == 2 and flag_yunits != 3:  # drawdown derivative data
            raise TypeError('Error in drawdown first derivative units {}'.
                            format(yunits))
        elif self.dtype == 3 and flag_yunits != 4:  # drawdown derivative data
            raise TypeError('Error in drawdown second derivative units {}'.
                            format(yunits))
        else:
            self.yunits = yunits
        # End Function

    def set_parameters(self, new_params):
        """
        Replaces the old parameters associated to data
        """
        self.parameters = _deepcopy(new_params)

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
                           'xunits': self.xunits,
                           'yunits': self.yunits,
                           'x': list(self.x),
                           'y': list(self.y),
                           'parameters': _deepcopy(self.parameters),
                           'plot_options': _deepcopy(self._graph)}
        return(data_properties)

    def to_file(self, filename, delimiter=','):
        """
        Exports the data to a delimiter file

        INPUTS:
          filename     [string] output delimited file (.txt, .csv)
          delimiter    [string] delimiter, by default ',' is used
        """
        if type(filename) is not str:
            raise TypeError("filename must be a string, a <{}> input".format(type(filename)))
        delim = delimiter
        with open(filename, 'w') as fout:
            fout.write('Name{}{}\n'.format(delim, self.name))
            fout.write('Description{}{}\n'.format(delim, self.description))
            fout.write('Data type{}{}\n'.format(delim, self.get_data_type()))
            fout.write('X units{}{}\n'.format(delim, self.xunits))
            fout.write('Y units{}{}\n'.format(delim, self.yunits))
            fout.write('Parameters\n')
            for key, value in self.parameters.items():
                fout.write('{}{}{}\n'.format(key, delim, self.xunits))
            fout.write('\nX{}Y'.format(delim))
            for i in range(self.x.size):
                fout.write('\n{}{}{}'.format(self.x[i], delim, self.y[i]))

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
        keys = ('_graph')
        for key in keys:
            fixed[key] = new_data.get(key, original.get(key))
        if type(fixed["x"]) is not _np.ndarray:
            fixed["x"] = _np.array(fixed["x"], dtype=_np.float32)
        if type(fixed["y"]) is not _np.ndarray:
            fixed["y"] = _np.array(fixed["y"], dtype=_np.float32)
        self.__dict__.update(fixed)

