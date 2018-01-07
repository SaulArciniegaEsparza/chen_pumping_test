"""
CHEN pumping test analysis
Pumping well and observation well properties
"""

import numpy as _np

# Create well data class
class _well_data(object):
    def __init__(self, dtype="drawdown", description="", name="data", color='k', symbol="", line="-"):
        self.X = _np.array([], dtype=_np.float32)  # time
        self.Y = _np.array([], dtype=_np.float32)  # depletion
        self.parameters = {}  # parameters
        self.dtype = dtype  # data type
        self.description = description  # data description
        self.name = name  # data name
        self.color = color  # data color (plot)
        self.symbol = symbol  # data symbol (plot)
        self.line = line  # data line (plot)
        self.visible = True  # data visibility (plot)

    # Make iterable object
    def __iter__(self):
        yield 'X', self.X
        yield 'Y', self.Y
        yield 'params', self.parameters
        yield 'dtype', self.dtype
        yield 'description', self.description
        yield 'name', self.name
        yield 'color', self.color
        yield 'symbol', self.symbol
        yield 'line', self.line
        yield 'visible', self.visible

    # Update data
    def update_data(self, new_data):
        self.__dict__.update(new_data)

    # Define getitem
    def __getitem__(self, key):
        obj = self.__dict__
        return(obj[key])

