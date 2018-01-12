"""
CHEN pumping test analysis
Aquifer properties initialization


Autor:
Saul Arciniega Esparza
zaul.ae@gmail.com
Institute of Engineering of UNAM
Mexico City, Mexico
"""

import wells as _wells


"""____________________________ WELL OBJECTS _______________________________"""


class Aquifer(object):
    def __init__(self, name="Aquifer", description="", time_units="s",
                 len_units="m", pump_units="m3/s", **kwargs):
        # Set attributes
        self._type = 0  # object identifier
        self.atype = 0  # aquifer identifier

        self.name = name
        self.description = description
        self.time_units = time_units
        self.len_units = len_units
        self.pump_units = pump_units

        # Set aquifer parameters
        self.parameters = {'T': kwargs.get('T', None),  # Transmisivity
                           'S': kwargs.get('S', None),  # Specific yield
                           'K': kwargs.get('K', None),  # Horizontal hydraulic conductivity (Kh)
                           'b': kwargs.get('b', None),  # Aquifer thickness
                           'Kzr': kwargs.get('Kzr', None)}  # Vertical anisotropy (Kzr=Kv/Kh)

        # Associated wells list
        self.wells = []

    def __str__(self):
        if self.atype == 0:
            text = '__________________Confined Aquifer___________________\n'
        elif self.atype == 1:
            text = '____________________Leaky Aquifer____________________\n'
        elif self.atype == 2:
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
        text += '        Specific yield S: {}'.format(self.parameters.get('S', None))
        text += '     Aquifer Thickness b: {}  {}'.format(self.parameters.get('b', None), self.len_units)
        text += ' Vertical anisotropy Kzr: {}'.format(self.parameters.get('Kzr', None))
        if self.atype == 1:
            text += 'Aquitard conductivity Kl: {}  {}'.format(self.parameters.get('Kl', None),
                                                              '%s/%s' % (self.len_units, self.time_units))
            text += 'Aquitard conductivity Kl: {}  {}'.format(self.parameters.get('Kl', None),
                                                              '%s/%s' % (self.len_units, self.time_units))
        if self.atype == 2:
            pass
        return(text)

    def __getitem__(self, key):
        return(self.__dict__.get(key, None))

    def __repr__(self):
        if self.atype == 0:
            raise('ConfinedAquifer(name={})'.format(self.name))
        elif self.atype == 1:
            raise('LeakyAquifer(name={})'.format(self.name))
        elif self.atype == 2:
            raise('UnconfinedAquifer(name={})'.format(self.name))
        else:
            return('ErrorElement!')

    def add_well(self):
        pass

    def convert_units(self):
        pass

    def delete_well(self):
        pass

    def delete_all_wells(self):
        pass

    def get_well_id(self):
        pass

    def get_well_name(self):
        pass

    def well_list(self):
        pass

    def set_parameters(self):
        pass

    def to_dict(self):
        pass

    def to_model(self):
        pass

    def update(self):
        pass

    def validate_parameters(self):
        pass

    def well_count(self):
        return(len(self.wells))


class ConfinedAquifer(Aquifer):
    def __init__(self):
        self.atype = 0  # confined aquifer id


class LeakyAquifer(Aquifer):
    def __init__(self, **kwargs):
        self.atype = 1  # leaky aquifer id
        self.__dict__update({'Kl': kwargs.get('Kl', None),  # aquitard conductivity
                             'bl': kwargs.get('bl', None)   # aquitard thickness
                            })


class UnconfinedAquifer(Aquifer):
    def __init__(self):
        self.atype = 2  # unconfined aquifer id