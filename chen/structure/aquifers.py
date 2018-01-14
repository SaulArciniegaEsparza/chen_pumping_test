"""
CHEN pumping test analysis
Aquifer properties initialization


Author:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City, Mexico
"""

from copy import deepcopy as _deepcopy
import wells as _wells


"""____________________________ WELL OBJECTS _______________________________"""


# Define default aquifer parameters
AQ_PARAMS = {'T': None,
             'K': None,
             'S': None,
             'b': 10.,
             'Kzr:': 10.}


class Aquifer(object):
    def __init__(self, name="Aquifer", description="", time_units="s",
                 len_units="m", pump_units="m3/s", **kwargs):
        """
        Creates an aquifer object that could contains one or more pumping wells

        INPUTS:
            name            [string] aquifer name
            description     [string] aquifer description
            time_units      [string] time units used
            len_units       [string] length units used
            pump_units      [string] pumping rate units used
            **kwargs        [dict] aquifer parameters that can contain
                              'T'    aquifer transmisivity
                              'S'    aquifer specific yield
                              'K'    aquifer hydraulic conductivity
                              'b'    aquifer thickness
                              'Kzr'  vertical anisotropy Krz=Kv/K
        ATTRIBUTES:
            atype           [int] aquifer identifier
                             0  undefined aquifer (default)
                             1  confined aquifer
                             2  leaky aquifer
                             3  unconfined aquifer
        Aquifer type is defined when a model is fitted or evaluated
        """

        # Set attributes
        self._type = 0  # object identifier
        self.atype = 0  # aquifer identifier

        self.name = name
        self.description = description
        self.time_units = time_units
        self.len_units = len_units
        self.pump_units = pump_units

        # Set aquifer parameters
        self.parameters = {'T': kwargs.get('T', AQ_PARAMS['T']),  # Transmisivity
                           'S': kwargs.get('S', AQ_PARAMS['S']),  # Specific yield
                           'K': kwargs.get('K', AQ_PARAMS['K']),  # Horizontal hydraulic conductivity (Kh)
                           'b': kwargs.get('b', AQ_PARAMS['b']),  # Aquifer thickness
                           'Kzr': kwargs.get('Kzr', AQ_PARAMS['Kzr'])}  # Vertical anisotropy (Kzr=Kv/Kh)

        # Associated wells and data list
        self.wells = []  # wells objects
        self.data = []   # model results

    def __str__(self):
        if self.atype == 0:
            text = '_______________________Aquifer______________________\n'
        elif self.atype == 1:
            text = '__________________Confined Aquifer___________________\n'
        elif self.atype == 2:
            text = '____________________Leaky Aquifer____________________\n'
        elif self.atype == 3:
            text = '_________________Unconfined Aquifer__________________\n'
        text += '   Aquifer code: {}\n'.format(self.atype)
        text += '           Name: {}\n'.format(self.name)
        text += '    Description: {}\n'.format(self.description)
        text += '     Time units: {}\n'.format(self.time_units)
        text += '   Length units: {}\n'.format(self.len_units)
        text += '  Pumping units: {}\n'.format(self.pump_units)
        text += 'No Assoc. wells: {}\n'.format(self.well_count())
        text += '\n___________________________________Aquifer parameters\n'
        text += '         Transmisivity T: {}  {}\n'.format(self.parameters.get('T', None),
                                                            '%s2/%s' % (self.len_units, self.time_units))
        text += '        Hydraulic Cond K: {}  {}\n'.format(self.parameters.get('K', None),
                                                            '%s/%s' % (self.len_units, self.time_units))
        text += '        Specific yield S: {}\n'.format(self.parameters.get('S', None))
        text += '     Aquifer Thickness b: {}  {}\n'.format(self.parameters.get('b', None), self.len_units)
        text += ' Vertical anisotropy Kzr: {}\n'.format(self.parameters.get('Kzr', None))
        for key, value in self.parameters.items():
            if key not in AQ_PARAMS.keys():
                text += '      Model parameter {}: {}\n'.format(key, value)
        return(text)

    def __getitem__(self, key):
        return(self.__dict__.get(key, None))

    def __repr__(self):
        if self.atype == 1:
            return('ConfinedAquifer(name={})'.format(self.name))
        elif self.atype == 2:
            return('LeakyAquifer(name={})'.format(self.name))
        elif self.atype == 3:
            return('UnconfinedAquifer(name={})'.format(self.name))
        else:
            return ('UndefinedAquifer(name={})'.format(self.name))

    def add_well(self, x=1, y=1, name="New well", description="Added well"):
        """
        Add new pumping well object to the actual aquifer

        INPUTS
         x            [int, float, list, tuple, ndarray] time vector
         y            [int, float, list, tuple, ndarray] drawodwn vector
         name         [string] well name that is used as label for plot
         description  [string] well description
        """
        new_well = _wells.PumpingWell(name=name, description=description, time_units=self.time_units,
                                      len_units=self.len_units, pump_units=self.pump_units)
        new_well.pumprate.set_data(x=x, y=y)
        self.wells.append(new_well)

    def convert_units(self):
        pass

    def delete_well(self, key):
        """
        Removes the well object from the associated wells list
        given the well's name or index
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
        Removes all the associated pumping wells
        """
        self.wells = []

    def get_well_id(self, name):
        """
        Returns the well id given a well name
        Only the first well with similar name is returned
        When well's name is not found, -1 is returned
        """
        idx = -1
        if type(name) is str:
            wells_names = self.wells_list()
            if name in wells_names:
                idx = wells_names.index(name)
        return(idx)

    def get_well_name(self, idx):
        """
        Returns well's name using the data index as input
        If well index does not exist then None is returned
        """
        n = self.well_count()
        assert 0 <= idx <= n - 1, "Bad well index"
        name = self.wells[idx].pumprate.name
        return(name)

    def set_parameters(self, T=None, S=None, K=None, b=None, Krz=None):
        """
        Set aquifer parameters
        When an error in input parameter is found, old values are preserved

        INPUTS:
         T      [float] transmisivity
         S      [float] specific yield
         K      [float] hydraulic conductivity
         b      [float] aquifer thickness
         Krz    [float] vertical anisotropy

         NOTE: if T is None and K and b are not None, T is estimated as T=K*b
         If K is None and T and b are not None, K is estimated as K=T/b
        """
        original = _deepcopy(self.parameters)  # save in case of error
        in_parameters = {'T': T,
                         'S': S,
                         'K': K,
                         'b': b,
                         'Krz': Krz}
        for key, value in in_parameters.items():
            if value is not None:
                self.parameters[key] = float(value)
        flag, warnings = self.validate_parameters()
        if not flag:
            self.parameters.update(original)
        if (self.parameters.get('T', None) is None and (self.parameters.get('K', None) is not None and
                                                        self.parameters.get('b', None) is not None)):
            self.parameters['T'] = self.parameters['K'] * self.parameters['b']
        if (self.parameters.get('K', None) is None and (self.parameters.get('T', None) is not None and
                                                        self.parameters.get('b', None) is not None)):
            self.parameters['K'] = self.parameters['T'] / self.parameters['b']
        # End Function

    def to_dict(self):
        """
        Returns a dictionary with all the data contained by the aquifer
        (including pumping wells). It could be used to save data into a .json file
        """
        out_dict = _deepcopy(self.__dict__)
        out_wells = []
        for i in range(self.well_count()):
            out_wells.append(self.wells[i].to_dict())
        out_dict["wells"] = out_wells
        return(out_dict)

    def to_model(self):
        """
        Returns a dictionary with aquifer parameters needed in models
        """
        out_dict = _deepcopy(self.parameters)
        out_dict["atype"] = self.atype
        return(out_dict)

    def to_file(self, filename):
        if type(filename) is not str:
            raise TypeError('filename must be a string')
        with open(filename, 'w') as fout:
            fout.write('###################CHEN PUMPING TEST######################\n\n')
            fout.write(str(self))
            for i in range(self.well_count()):
                fout.write('__________________________________________Pumping Well\n\n')
                fout.write(str(self.wells[i]) + '\n\n')
                for j in range(self.wells[i].well_count()):
                    fout.write(str(self.wells[i].wells[j]) + '\n\n')
                    for k in range(self.wells[i].wells[j].data_count()):
                        fout.write(str(self.wells[i].wells[j].data[k]) + '\n\n')
        # End Function

    def update(self, new_data):
        """
        Updates the aquifer object using an input dictionary
        """
        if type(new_data) is not dict:
            raise TypeError("Input parameter must be a dict")
        # Update parameters
        self._type = new_data.get("_type", self._type)
        self.time_units = new_data.get("time_units", self.time_units)
        self.len_units = new_data.get("len_units", self.len_units)
        self.pump_units = new_data.get("pump_units", self.pump_units)
        self.parameters = new_data.get("parameters", self.parameters)
        # Update pumping wells
        if "wells" in new_data:
            n = len(new_data["wells"])
            if n > 1:
                self.delete_all_wells()
                for i in range(n):
                    self.add_well()
                    self.wells[i].update(new_data["wells"][i])
        # End Function

    def validate_parameters(self):
        """
        Verify aquifer parameters and returns warnings according to
        possible errors

        OUTPUTS:
         flag       [bool] if an error is detected in parameters
                      then flag is returned as False, in other way True
         warnings   [string] warnings text
        """
        flag = True
        warnings = ""
        for key in AQ_PARAMS.keys():
            value = self.parameters.get(key, None)
            if value is None:
                warnings += "Parameter {} has been defined for optimization\n".format(key)
            elif type(value) not in [int, float]:
                flag = False
                warnings += "Parameter {} must be a float value\n".format(key)
            else:
                if value <= 0:
                    flag = False
                    warnings += "Parameter {} must be higher than 0\n".format(key)
            return(flag, warnings)

    def well_count(self):
        """
        Returns the number of associated wells
        """
        return(len(self.wells))

    def wells_list(self):
        """
        Returns a list with the wells' name
        """
        list_names = []
        for well_data in self.wells:
            list_names.append(well_data.pumprate.name)
        return(list_names)

