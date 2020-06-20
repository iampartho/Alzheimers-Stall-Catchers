from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


class VoxelFilter:

    def __init__(self):
        self.data = []

    def low_pass_filter(self, input_signal):

        sos = signal.butter(2, 100, 'lp', fs=1000, output='sos')
        filtered = signal.sosfilt(sos, input_signal)

        return filtered
