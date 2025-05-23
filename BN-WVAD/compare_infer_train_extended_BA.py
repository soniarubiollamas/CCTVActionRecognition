import os
import numpy as np
import matplotlib.pyplot as plt

# Function to plot .npy files with ground truth and predictions
def plot_npy_files_with_gt(gt_path, pel4vad_path, context_path, filename, ax, crop_context=False):
    # Load the data
    gt = np.load(gt_path)
    pel4vad_data = np.load(pel4vad_path)
    context_data = np.load(context_path)

    if crop_context:
        # Crop the context data to represent only the part with the original frames
        max_len = max(len(pel4vad_data), len(gt))
        context_len = (len(context_data) - len(gt)) // 2
        context_data = context_data[context_len:len(context_data)-context_len]
    else:
        # Adjust sizes to represent the whole context
        max_len = max(len(pel4vad_data), len(context_data))
        context_len = (len(context_data) - len(gt)) // 2
        pel4vad_data = np.append(np.repeat(0, context_len), pel4vad_data)
        pel4vad_data = np.append(pel4vad_data, np.repeat(0, context_len))
        if max_len < len(gt):
            # Truncate gt
            gt = gt[:len(context_data)]

        if max_len > len(gt):
            gt = np.append(np.repeat(gt[0], context_len), gt)
            gt = np.append(gt, np.repeat(gt[-1], context_len))

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

    # Plot the PEL4VAD predictions
    ax.plot(pel4vad_data, label="PEL4VAD (w/o context)", linestyle="-", color="blue")

    # Plot the predictions with context
    ax.plot(context_data, label="PEL4VAD (w/ context)", linestyle="-", color="cyan")

    ax.set_xlabel("Frames")
    ax.set_ylabel("Value")
    filename = filename.split("_")[0]
    ax.set_title(f"{filename} Prediction")

    # Adjust plot limits
    ax.set_ylim(0, 1.01)
    ax.set_xlim(0, len(gt))

    ax.legend()

# Parameters
mode = "test"  # Can be "test" or "train"
name = "Vandalism028"  # File name to process
filename = f"{name}_pred.npy"

# Paths
base_path = "C:/Users/xoni/Documents/1_uni/1_insterships/"
gt_path = f"{base_path}cuda_version/PEL4VAD-master/predictions/gt/{name}_x264_pred.npy"
pel4vad_path = f"{base_path}BN-WVAD-main/frame_label/predictions/test_normalized/30fps/{filename}"
context_path = f"{base_path}BN-WVAD-main/frame_label/predictions/test_normalized/30fps/context/{filename}"

# Create the plot with a 2x1 grid
fig, axes = plt.subplots(2, 1, figsize=(10, 12))



# Plot with cropped context in the second graphic
plot_npy_files_with_gt(gt_path, pel4vad_path, context_path, filename, axes[0], crop_context=True)

# Plot with full context in the first graphic
plot_npy_files_with_gt(gt_path, pel4vad_path, context_path, filename, axes[1], crop_context=False)

# Show the plots
plt.tight_layout()
plt.show()