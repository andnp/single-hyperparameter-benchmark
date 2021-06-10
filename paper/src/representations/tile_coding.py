from PyFixedReps import TileCoder
from PyExpUtils.utils.dict import merge

class MyTileCoder(TileCoder):
    def __init__(self, params, dims):
        super().__init__(merge(params, { 'dims': dims }))

    def encode(self, s, a=None):
        return super().get_indices(s, a)
