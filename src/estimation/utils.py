import numpy as np

def calculate_roughness(data):
    second_diffs = np.diff(np.diff(data, axis=1), axis=1)
    smoothness = np.abs(second_diffs).mean()

    return float(smoothness)