import os
import numpy as np
import matplotlib.pyplot as plt

# Function to plot an .npy file with ground truth and a single algorithm
def plot_npy_files_with_gt(gt_path, algo_path, filename, algo_label, algo_color, ax):
    # Load the data
    gt = np.load(gt_path)
    algo_data = np.load(algo_path)

    # Identify anomaly regions in the ground truth
    anomaly_frames = np.where(gt > 0.5)[0]  # Assuming anomalies are labeled as > 0.5
    if len(anomaly_frames) > 0:
        ax.fill_between(
            range(len(gt)),
            0,
            1,
            where=(gt > 0.5),
            color="red",
            alpha=0.3,
            label="Ground Truth",
        )

    # Plot the algorithm's predictions
    ax.plot(algo_data, label=algo_label, linestyle="-", color=algo_color)

    ax.set_xlabel("Frame Number")
    ax.set_ylabel("p(A)")
    filename = filename.split("_")[0]
    ax.set_title(f"{filename} Prediction")

    # Adjust plot limits
    ax.set_ylim(0, 1.01)
    ax.set_xlim(0, len(gt))

    ax.legend()

# Parameters
mode = "test"  # Can be "test" or "train"
algorithm = "PEL4VAD"  # Select the algorithm: "PEL4VAD", "UR-DMU", or "BN-WVAD"
name = "Explosion016"  # File name to process
filename = f"{name}_pred.npy"

# Paths
base_path = "C:/Users/xoni/Documents/1_uni/1_insterships/"
gt_path = f"{base_path}cuda_version/PEL4VAD-master/predictions/gt/{name}_x264_pred.npy"

if algorithm == "PEL4VAD":
    algo_path = f"{base_path}cuda_version/PEL4VAD-master/predictions/{mode}/{filename}"
    algo_label = "PEL4VAD"
    algo_color = "blue"
elif algorithm == "UR-DMU":
    algo_path = f"{base_path}UR-DMU/frame_label/predictions/{mode}/30fps/{filename}"
    algo_label = "UR-DMU"
    algo_color = "green"
elif algorithm == "BN-WVAD":
    algo_path = f"frame_label/predictions/{mode}_normalized/30fps/{filename}"
    algo_label = "BN-WVAD"
    algo_color = "yellow"
else:
    raise ValueError("Unrecognized algorithm. Select 'PEL4VAD', 'UR-DMU', or 'BN-WVAD'.")

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
plot_npy_files_with_gt(gt_path, algo_path, filename, algo_label, algo_color, ax)

# Show the plot
plt.tight_layout()
plt.show()
