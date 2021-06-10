import numpy as np
from representations.tile_coding import MyTileCoder
from PyFixedReps import BaseRepresentation

class ScaleRep(BaseRepresentation):
    def __init__(self, params, dims):
        super().__init__()
        self.feats = dims

    def features(self):
        return self.feats

    def encode(self, s, a=None):
        s = 2.0 * np.array(s) - 1.0
        return s

class IdentityRep(BaseRepresentation):
    def __init__(self, params, dims):
        super().__init__()
        self.dims = dims

    def features(self):
        return self.dims

    def encode(self, s, a=None):
        return s

class LLTileCoder(MyTileCoder):
    def __init__(self, params, dims):
        super().__init__(params, 7)

    def encode(self, s, a=None):
        # first 6 features should be tile-coded together
        s_ = s[0:6]
        # last two should be done separately
        a = int(2 * s[6] + s[7])

        return super().encode(np.append(s_, a))


def getRepresentation(name):
    if name == 'tile-coder':
        return MyTileCoder

    if name == 'scale':
        return ScaleRep

    if name == 'll-tile-coder':
        return LLTileCoder

    return IdentityRep
