import random

def getControlledSeed(controlled: bool, seed: int):
    if controlled:
        return seed

    return random.randint(0, 1000000)
