import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Birdseye:
    def __init__(self, data):
        self.data = data

    def plot(self):
        plt.plot(self.data)
        plt.show()