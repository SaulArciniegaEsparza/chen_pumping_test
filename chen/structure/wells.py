"""
CHEN pumping test analysis
Pumping well and observation well properties

CLASSES:
    PumpingWell       > Pumping well object that could contain one or more Observation wells or piezometers
    ObservationWell   > Observation well that contains drawdown data
    Piezometer        > Piezometer well that contains drawdown data
    WellData          > Object used to store different data (pumping rate, drawdown, first and second derivative)

See help for more information about how to work with each object
PumpingWell is the main object, so it would be created in first instance



Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City, Mexico
"""

from copy import deepcopy as _deepcopy
from data import Data as _Data


"""____________________________ WELL OBJECTS _______________________________"""


class PumpingWell(object):
    def __init__(self, name="", description="", time_units="s", len_units="m",
                 pump_units="m3/s"):
        """
        Create a pumping rate well

        ATTRIBUTES:
            name:         [string] pumping well name that is used in plots
            description:  [string] short well description
            time_units:   [string] time units for this well and associated data
            len_units:    [string] length units for this well and associated data
            pump_units:   [string] pumping rate units
            parameters:   dictionary that contains the well parameters
                 rw       [float] well radius in longitude units
                 d        [float] depth of well screen (from top) in length units
                 l        [float] depth of well bottom screen in length units
            pumprate:     data object that contains the drawdown data
            wells:        [list] contains data objects with the results of applied models

        Creating a new well:
        well = PumpingWell(name="Well 1", description="First well added", time_units="s",
                           len_units="m", pump_units="m3/s")
        well.pumprate.set_data(x, y)
        well.set_parameters(rw=0.8, d=5., l=15., full=False)

        Optionally:
        well = PumpingWell(name="Well 1", description="First well added", time_units="s",
                           len_units="m", pump_units="m3/s")
        well.pumprate.set_data(x, y)
        well.set_parameters(rw=0.8, d=5., l=15., full=False)

        Adding new observation well:
        well.add_well(x, y, wtype=0, name="New well", description="Added well")
        Adding new piezometer:
        well.add_well(x, y, wtype=1, name="New Piezometer", description="Added piezometer")
        """

        # Set general info
        self._type = 1  # pumping well id
        self.parameters = {'full': True,
                           'rw': 1.,
                           'd':  0.,
                           'l':  1.}
        self.time_units = time_units
        self.len_units = len_units
        self.pump_units = pump_units

        # Create pumping well data
        self.pumprate = _Data(dtype=0, name=name, description=description)
        self.pumprate.set_units(self.time_units, self.pump_units)

        # Set observation wells and piezometers
        self.wells = []

    def __getitem__(self, key):
        return(self.__dict__[key])

    def __repr__(self):
        return("PumpingWell(name='{}')".format(self.pumprate.name))

    def __str__(self):
        text = "_____________Pumping well_______________\n\n"
        text += "    Well type: {}\n".format(self._type)
        text += "    Well name: {}\n".format(self.pumprate.name)
        text += "  Description: {}\n".format(self.pumprate.description)
        text += "   Time Units: {}\n".format(self.time_units)
        text += " Length Units: {}\n".format(self.len_units)
        text += "PumpRate Units: {}\n".format(self.pump_units)
        text += "No Assoc Wells: {}\n".format(self.well_count())
        text += "\n______________Pumping rate attributes\n"
        text += "         Length: {}\n".format(len(self.pumprate.x))
        if self.is_constant_rate():
            text += "       Pumping: {}  {}\n".format(self.pumprate.y[0], self.pump_units)
        else:
            text += "      Star time: {}  {}\n".format(self.pumprate.x[0], self.time_units)
            text += "     Final time: {}  {}\n".format(self.pumprate.x[-1], self.time_units)
            text += "  Pumping range: {} - {}  {}\n".format(max(self.pumprate.y), min(self.pumprate.y),
                                                           self.pump_units)
        text += "\n_____________________Well attributes\n"
        text += "   Full penetration: {}\n".format(self.parameters["full"])
        text += "     Well radius rw: {} {}\n".format(self.parameters["rw"], self.len_units)
        if not self.parameters["full"]:
            text += "     Bottom depth l: {} {}\n".format(self.parameters["l"], self.len_units)
            text += " Top Screen depth d: {} {}\n".format(self.parameters["d"], self.len_units)
        return(text)

    def add_well(self, x=1, y=1, wtype=0, name="New well", description="Added well"):
        """
        Add new observation well or piezometer object to the actual pumping well

        INPUTS
         x            [int, float, list, tuple, ndarray] time vector
         y            [int, float, list, tuple, ndarray] drawodwn vector
         wtype        [int] type of data
                       0     observation well
                       1     piezometer
         name         [string] well name that is used as label for plot
         description  [string] well description
        """
        if wtype == 0:
            new_well = ObservationWell(name, description, time_units=self.time_units,
                                       len_units=self.len_units)
        elif wtype == 1:
            new_well = Piezometer(name, description)
        else:
            raise ValueError('Bad wtype value <{}>'.format(wtype))
        new_well.drawdown.set_data(x=x, y=y)
        self.wells.append(new_well)

    def convert_units(self, time_units=None, len_units=None, pump_units=None,
                      same=False):
        """
        Convert parameters, pumping rate, drawdown and data units given new units

        INPUTS
         time_units       [string] new time units. If None, actual units are used
         length_units     [string] new length units. If None, actual units are used
         pump_units       [string] new pumping rate units. If None, actual units are used
         same             [bool] if True, pump_units is created from input length and time units

         NOTE: pumping rate could have different time and len units.
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
        # Check new pumping rate units
        in_pump = self.pump_units
        if pump_units is None:
            pump_units = in_pump
        if same:
            pump_units = "%s3/%s" % (len_units, time_units)
        flag = _units.validate_units(pump_units)
        if flag == -1:
            raise ValueError('Bad pumping rate units input {}'.format(len_units))

        # Convert parameters units
        for key, value in self.parameters.items():
            if type(value) in [int, float]:
                self.parameters[key] = _units.units_conversion(value, in_len, len_units)
        # Convert pumping rate data
        self.pumprate.convert_units(time_units, pump_units)
        # Convert well data units
        for i in range(self.well_count()):
            self.wells[i].convert_units(time_units, len_units)
        # Set input units
        self.len_units = len_units
        self.time_units = time_units
        self.pump_units = pump_units
        # End Function

    def convert_same_units(self):
        """
        Converts the actual pumping rate units to the equivalent
        length and time units used in parameters and drawdown
        """
        # Convert pumping rate data
        pump_units = "%s3/%s" % (self.len_units, self.time_units)
        flag = _units.validate_units(pump_units)
        if flag == 2:
            self.pumprate.convert_units(self.time_units, pump_units)
            self.pump_units = pump_units

    def delete_well(self, key):
        """
        Removes the well object from the associated wells list
        given the well name or index
        """
        if type(key) is str:
            idx = self.get_well_id(key)
        elif type(key) is int:
            idx = key
        else:
            raise TypeError('key must be a string or an integer.')
        n = self.well_count()
        if 0 <= idx <= n - 1:
            raise ValueError('Bad value for key parameter')
        del(self.wells[idx])

    def delete_all_wells(self):
        """
        Removes all the associated observation wells or piezometers
        """
        self.wells = []

    def get_plot_options(self):
        """
        Returns a list of plot options with the visible data to be plotted
        including pumping well, observation well or piezometer drawdown
        """
        plot_options = []
        # Get pumping rate plot options
        op = self.pumprate.get_plot_options()
        if op['visible']:
            plot_options.append(op)
        # Get associated data options
        for i in range(self.well_count()):
            well_options = self.wells[i].get_plot_options()
            plot_options.extend(well_options)
        return(plot_options)

    def get_well_id(self, name):
        """
        Returns the well id given a well name
        Only the first well with similar names is returned
        When well name is not found, -1 is returned
        """

        idx = -1
        if type(name) is str:
            wells_names = self.wells_list()
            if name in wells_names:
                idx = wells_names.index(name)
        return(idx)

    def get_well_name(self, idx):
        """
        Returns well name using the data index as input
        If well index does not exist then None is returned
        """
        name = None
        if type(idx) is int:
            n = self.well_count()
            assert 0 <= idx <= n - 1, "Bad well index"
            name = self.wells[idx].drawdown.name
        return(name)

    def is_constant_rate(self):
        """
        Check if the well has a constant pumping rate (True) or if
        pumping rate varies in time (False)
        """
        n1 = self.pumprate.x.size
        n2 = self.pumprate.y.size
        if n1 == n2:
            if n1 == 1:
                return(True)
            else:
                return(False)
        else:
            raise ValueError('Pumping rate is incorrect, check the assigned values!')

    def set_parameters(self, full=None, rw=None, l=None, d=None):
        """
        Set well attributes

        INPUTS:
         full   [bool] if True, well is full penetrating and depth
                 parameters are ignored in computation
         rw     [float] well radius in length units
         l      [float] depth from water table to well bottom in length units
         d      [float] depth from water table to well top screen in length units
        """

        original = _deepcopy(self.parameters)  # save in case of error

        if type(full) is bool:
            self.parameters["full"] = full
        if type(rw) in (int, float):
            self.parameters["rw"] = float(rw)
        if type(d) in (int, float):
            self.parameters["d"] = float(d)
        if type(l) in (int, float):
            self.parameters["l"] = float(l)

        flag, message = self.validate_parameters()
        if not flag:
            print(message)
            self.parameters.update(original)
        # End Function

    def to_dict(self):
        """
        Returns a dictionary with all the data contained by the pumping well
        (including observation wells). It could be used to save data into a .json file
        """
        out_dict = _deepcopy(self.__dict__)
        out_dict["pumprate"] = self.pumprate.to_dict()
        out_wells = []
        for i in range(self.well_count()):
            out_wells.append(self.wells[i].to_dict())
        out_dict["wells"] = out_wells
        return(out_dict)

    def to_model(self):
        """
        Returns a dictionary with drawdown data and well parameters
        needed for analysis models in chen
        """
        out_dict = _deepcopy(self.parameters)
        out_dict["x"] = self.pumprate.x.copy()
        out_dict["y"] = self.pumprate.y.copy()
        out_dict["wtype"] = 1
        return(out_dict)

    def update(self, new_data):
        """
        Updates the pumping well object using an input dictionary
        """
        if type(new_data) is not dict:
            raise TypeError("Input parameter must be a dict")
        # Update parameters
        self._type = new_data.get("_type", self._type)
        self.time_units = new_data.get("time_units", self.time_units)
        self.len_units = new_data.get("len_units", self.len_units)
        self.pump_units = new_data.get("pump_units", self.pump_units)
        self.parameters = new_data.get("parameters", self.parameters)
        # Update pumping rate
        self.pumprate.update(new_data.get("pumprate", self.pumprate.to_dict()))
        # Update data
        if "wells" in new_data:
            n = len(new_data["wells"])
            if n > 1:
                self.delete_all_wells()
                for i in range(n):
                    self.add_well(0, 0, new_data["wells"][i]["_type"] - 2)
                    self.wells[i].update(new_data["wells"][i])
        # End Function

    def validate_parameters(self):
        """
        Verify well parameters and returns warnings according to
        possible errors

        OUTPUTS:
         flag       [bool] if an error is detected in parameters
                      then flag is returned as False, in other way True
         warnings   [string] warnings text
        """

        flag = True
        warnings = ""
        # Check radius
        r = self.parameters.get('rw', 0)
        if type(r) not in [int, float]:
            flag = False
            warnings += "Well radius rw must be a float value\n"
        else:
            if r <= 0:
                flag = False
                warnings += "Well radius rw must be higher than 0\n"
        # Check if is full penetrating
        op = self.parameters.get('full', False)

        if not op:
            # Check observation well length
            if 'd' in self.parameters and 'l' in self.parameters:
                d = self.parameters.get('d', -1)
                l = self.parameters.get('l', -1)
                if type(l) not in [int, float]:
                    flag = False
                    warnings += "Depth of well bottom must be a float value\n"
                else:
                    if l < 0:
                        flag = False
                        warnings += "Depth l must be higher than 0\n"
                if type(d) not in [int, float]:
                    flag = False
                    warnings += "Depth of well screen must be a float value\n"
                else:
                    if d < 0 or d > l:
                        flag = False
                        warnings += "Depth d must be in range 0 <= d <= l\n"
        return(flag, warnings)  # End Function

    def well_count(self):
        """
        Returns the number of observation wells associate to pumping well
        """
        return(len(self.wells))

    def wells_list(self, wtype='all'):
        """
        Returns a list with the well name given a type of well

        INPUTS:
        dtype       [string, int] type of data
                      'all'   All well names are returned
                       0      Only observation wells
                       1      Only piezometers
        """
        list_names = []
        for well_data in self.wells:
            if wtype == 'all':
                list_names.append(well_data.drawdown.name)
            elif wtype == well_data._type - 2:
                list_names.append(well_data.drawdown.name)
        return(list_names)


class ObservationWell(object):
    def __init__(self, name="", description="", time_units="s", len_units="m"):
        """
        Create an observation well object

        ATTRIBUTES:
            name:         [string] well name that is used in plots
            description:  [string] short well description
            time_units:   [string] time units for this well and associated data
            len_units:    [string] length units for this well and associated data
            parameters:   [dict] dictionary that contains the well parameters
                 r         [float] radial distance to pumping well in longitude units
                 d         [float] depth of well screen (from top) in length units
                 l         [float] depth of well bottom screen in length units
            drawdown:    data object that contains the drawdown data
            data:        [list] contains data objects with the results of applied models

        Creating a new well:
        well = ObservationWell(name='Well 1', description='First well added')
        well.drawdown.set_data(x, y, xunits='min', yunits='m')
        well.set_parameters(r=50., d=5., l=15., full=False)

        Optionally:
        well = ObservationWell(name='Well 1', description='First well added')
        well.drawdown.import_data_from_filefilename, delimiter=',', skip_header=1, xunits='min', yunits='m')
        well.set_parameters(r=50., d=5., l=15., full=False)

        Adding new data:
        well.add_data(x, model, dtype=1, name="Theis",
                      description="Theis method simulation")
        well.add_data(x, derivative, dtype=2, name="ds/dt Bourdet",
                      description="First derivative with Bourdet method")
        well.add_data(x, derivative, dtype=3, name="d2s/dt2 Bourdet",
                      description="Second derivative with Bourdet")
        """

        # Set general info
        self._type = 2  # observation well id
        self.time_units = time_units
        self.len_units = len_units

        self.parameters = {'full': True,  # is full penetrating?
                           'r': 1.,  # radius, distance until pumping well in length units
                           'd': 0.,  # depth of well screen (from top) in length units
                           'l': 1.}  # depth of well bottom in length units

        # Create drawdown data
        self.drawdown = _Data(dtype=1, name=name, description=description)
        self.drawdown.set_units(self.time_units, self.len_units)

        # Set results from models
        self.data = []

    def __getitem__(self, key):
        return(self.__dict__[key])

    def __repr__(self):
        if self._type == 2:
            return("ObservationWell(name='{}')".format(self.drawdown.name))
        else:
            return("Piezometer(name={})".format(self.drawdown.name))

    def __str__(self):
        if self._type == 2:
            text = "___________Observation well_____________\n\n"
        else:
            text = "______________Piezometer________________\n\n"
        text += "    Well type: {}\n".format(self._type)
        text += "    Well name: {}\n".format(self.drawdown.name)
        text += "  Description: {}\n".format(self.drawdown.description)
        text += "   Time Units: {}\n".format(self.time_units)
        text += " Length Units: {}\n".format(self.len_units)
        text += "No Assoc Data: {}\n".format(self.data_count())
        text += "\n_______________________Drawdown attributes\n"
        text += "         Length: {}\n".format(len(self.drawdown.x))
        text += "      Star time: {}  {}\n".format(self.drawdown.x[0], self.time_units)
        text += "     Final time: {}  {}\n".format(self.drawdown.x[-1], self.time_units)
        text += " Drawdown range: {} - {}  {}\n".format(max(self.drawdown.y), min(self.drawdown.y),
                                                        self.len_units)
        text += "\n___________________________Well attributes\n"
        text += " Full penetration: {}\n".format(self.parameters["full"])
        text += "       Distance r: {} {}\n".format(self.parameters["r"], self.len_units)
        if not self.parameters["full"]:
            if self._type == 2:
                text += "     Bottom depth l: {} {}\n".format(self.parameters["l"], self.len_units)
                text += " Top Screen depth d: {} {}\n".format(self.parameters["d"], self.len_units)
            else:
                text += " Piezometer depth z: {} {}\n".format(self.parameters["z"], self.len_units)
        return(text)

    def add_data(self, x=1, y=1, dtype=1, name="New data", description="New data"):
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
        new_data = _Data(dtype, name=name, description=description)
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
        self.len_units = len_units
        self.time_units = time_units
        # End Function

    def delete_data(self, key):
        """
        Removes the data object from the associated data list
        given the data name or id
        """
        if type(key) is str:
            idx = self.get_data_id(key)
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

    def data_list(self, dtype='all'):
        """
        Returns a list with the data name given a type of data

        INPUTS:
        dtype       [string, int] type of data
                      'all'   All data names are returned
                       1      Only drawdown data names are returned
                       2      Only first derivative drawdown
                       3      Only second derivative drawdown
        """
        list_names = []
        for well_data in self.data:
            if dtype == 'all':
                list_names.append(well_data.name)
            elif dtype == well_data.dtype:
                list_names.append(well_data.name)
        return(list_names)

    def data_count(self):
        """
        Returns the number of data associated to the well or piezometer
        """
        return(len(self.data))

    def get_data(self, key):
        """
        Returns the data object giving the data name or index
        """
        if type(key) is str:
            idx = self.get_data_id(key)
        elif type(key) is int:
            idx = key
        else:
            raise TypeError('key must be a string or a integer.')
        n = self.data_count()
        if 0 > idx or idx > n - 1:
            raise ValueError('Bad value for key parameter')
        return(self.data[idx])

    def get_data_id(self, name):
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

    def get_data_name(self, idx):
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

    def get_data_type(self, idx):
        """
        Returns the data type given the data index
        """
        return(self.data[idx].dtype)

    def get_parameters(self):
        """
        Returns a dictionary with well parameters
        """
        return(_deepcopy(self.parameters))

    def reset_data(self):
        """
        Delete all the associate data
        """
        self.data = []

    def set_as_drawdown(self, idx):
        """
        Replaces the well drawdown given the index of a drawdown data
        in data list
        """
        dtype = self.get_data_type(idx)
        if dtype == "Pumping rate":
            x, y = self.data[idx].get_data()
            self.drawdown.set_data(x, y)
        else:
            raise TypeError('Selected data is not drawdown!')

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
        if not flag:
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

    def to_model(self):
        """
        Returns a dictionary with drawdown data and well parameters
        needed for analysis models in chen
        """
        out_dict = _deepcopy(self.parameters)
        out_dict["x"] = self.drawdown.x.copy()
        out_dict["y"] = self.drawdown.y.copy()
        if self._type == 2:
            out_dict["wtype"] = 2
        elif self._type == 3:
            out_dict["wtype"] = 3
        return(out_dict)

    def update(self, new_data):
        """
        Updates the well or piezometer object using an input dictionary
        """
        if type(new_data) is not dict:
            raise TypeError("Input parameter must be a dict")
        # Update parameters
        self._type = new_data.get("_type", self._type)
        self.time_units = new_data.get("time_units", self.time_units)
        self.len_units = new_data.get("len_units", self.len_units)
        self.parameters = new_data.get("parameters", self.parameters)
        # Update drawdown
        self.drawdown.update(new_data.get("drawdown", self.drawdown.to_dict()))
        # Update data
        if "data" in new_data:
            n = len(new_data["data"])
            if n > 0:
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
         flag       [bool] if an error is detected in parameters
                      then flag is returned as False, in other way True
         warnings   [string] warnings text
        """

        flag = True
        warnings = ""
        # Check radius
        r = self.parameters.get('r', 0)
        if type(r) not in [int, float]:
            flag = False
            warnings += "Radius r must be a float value\n"
        else:
            if r <= 0:
                flag = False
                warnings += "Radius r must be higher than 0\n"
        # Check if is full penetrating
        op = self.parameters.get('full', False)

        if not op:
            # Check observation well length
            if 'd' in self.parameters and 'l' in self.parameters:
                d = self.parameters.get('d', -1)
                l = self.parameters.get('l', -1)
                if type(l) not in [int, float]:
                    flag = False
                    warnings += "Depth of well bottom must be a float value\n"
                else:
                    if l < 0:
                        flag = False
                        warnings += "Depth l must be higher than 0\n"
                if type(d) not in [int, float]:
                    flag = False
                    warnings += "Depth of well screen must be a float value\n"
                else:
                    if d < 0 or d > l:
                        flag = False
                        warnings += "Depth d must be in range 0 <= d <= l\n"
            # Check piezometer depth
            elif 'z' in self.parameters:
                z = self.parameters.get('z', -1)
                if type(z) not in [int, float]:
                    flag = False
                    warnings += "Depth of piezometer must be a float value\n"
                else:
                    if z < 0:
                        flag = False
                        warnings += "Depth z must be higher than 0\n"
            else:
                flag = False
                warnings += "Well don't contain well depth attributes\n"
        return(flag, warnings)  # End Function


class Piezometer(ObservationWell):
    def __init__(self, name="", description="", time_units="s", len_units="m"):
        """
        Create a piezometer object (works similar to Observation well)

        ATTRIBUTES:
            name:         [string] piezometer name that is used in plots
            description:  [string] short piezometer description
            time_units:   [string] time units for this piezometer and associated data
            len_units:    [string] length units for this piezometer and associated data
            parameters:   [dict] dictionary that contains the piezometer parameters
                 r         [float] radial distance to pumping well in longitude units
                 z         [float] piezometer depth in length units
            drawdown:    data object that contains the drawdown data
            data:        [list] contains data objects with the results of applied models

        Creating a new well:
        well = Piezometer(name="Piezometer 1", description="First piezometer added",
                          time_units="s", len_units="m")
        well.drawdown.set_data(x, y)
        well.set_parameters(r=50., z=15., full=False)

        Optionally:
        well = Piezometer(name="Piezometer 1", description="First piezometer added",
                          time_units="s", len_units="m")
        well.drawdown.from_file(filename, delimiter=',', skip_header=1)
        well.set_parameters(r=50., d=5., l=15., full=False)

        Adding new data:
        well.add_data(x, model, dtype=1, name="Theis",
                      description="Theis method simulation")
        well.add_data(x, derivative, dtype=2, name="ds/dt Bourdet",
                      description="First derivative with Bourdet method")
        well.add_data(x, derivative, dtype=3, name="d2s/dt2 Bourdet",
                      description="Second derivative with Bourdet")
        """
        super(Piezometer, self).__init__()

        # Set general info
        self._type = 3  # piezometer id
        self.parameters = {'full': True,  # is full penetrating?
                           'r': 1.,  # distance until pumping well in length units
                           'z': 1.}  # piezometer depth in length units

        # Set data
        self.name = name
        self.description = description

        self.time_units = time_units
        self.len_units = len_units

        # Create drawdown data
        self.drawdown = _Data(dtype=1, name=name, description=description)
        self.drawdown.set_units(self.time_units, self.len_units)

