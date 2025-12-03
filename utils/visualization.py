import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_motion(motion, save_path):
    """
    Plots a sequence of poses.
    Args:
        motion: [T, D] numpy array.
        save_path: path to save the plot.
    """
    # Placeholder: Just plot the first 3 dimensions as a trajectory for now
    # Real implementation would need skeleton connectivity
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Assuming first 3 dims are root position
    if motion.shape[1] >= 3:
        ax.plot(motion[:, 0], motion[:, 1], motion[:, 2])
    
    plt.savefig(save_path)
    plt.close()
