import numpy as np
import matplotlib.pyplot as plt

# Function to plot 3 algorithms with ground truth shadowed
def plot_with_downsampled_shadows_3x1(folder_original1, folder_original2, folder_original3, folder_downsampled1, folder_downsampled2, folder_downsampled3, folder_groundtruth, filename, label1, label2, label3):
    # Load the data
    data_original1 = np.load(folder_original1)
    data_original2 = np.load(folder_original2)
    data_original3 = np.load(folder_original3)

    data_downsampled1 = np.load(folder_downsampled1)
    data_downsampled2 = np.load(folder_downsampled2)
    data_downsampled3 = np.load(folder_downsampled3)

    # Load the ground truth
    data_groundtruth = np.load(folder_groundtruth)

    # Create figure and subplots for a 3x1 grid
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    
    # Helper function to add ground truth shadow
    def add_ground_truth_shadow(ax, data_groundtruth):
        anomaly_frames = np.where(data_groundtruth > 0.5)[0]  # Assuming anomalies are > 0.5
        if len(anomaly_frames) > 0:
            ax.fill_between(
                range(len(data_groundtruth)),
                0,
                1,
                where=(data_groundtruth > 0.5),
                color="red",
                alpha=0.3,
                label="Ground Truth"
            )

    # Plot for the first algorithm
    add_ground_truth_shadow(ax[0], data_groundtruth)
    ax[0].plot(data_original1, label=f"{label1} Original", linestyle='-', color='blue')
    ax[0].plot(data_downsampled1, label=f"{label1} Downsampled", linestyle='--', color='blue')
    ax[0].set_xlabel("Frames")
    ax[0].set_ylabel("Value")
    ax[0].set_title(f"{label1}")
    ax[0].set_ylim(0, 1.03)
    ax[0].set_xlim(0, max(len(data_original1), len(data_downsampled1)))
    ax[0].legend()

    # Plot for the second algorithm
    add_ground_truth_shadow(ax[1], data_groundtruth)
    ax[1].plot(data_original2, label=f"{label2} Original", linestyle='-', color='green')
    ax[1].plot(data_downsampled2, label=f"{label2} Downsampled", linestyle='--', color='green')
    ax[1].set_xlabel("Frames")
    ax[1].set_ylabel("Value")
    ax[1].set_title(f"{label2}")
    ax[1].set_ylim(0, 1.03)
    ax[1].set_xlim(0, max(len(data_original2), len(data_downsampled2)))
    ax[1].legend()

    # Plot for the third algorithm
    add_ground_truth_shadow(ax[2], data_groundtruth)
    ax[2].plot(data_original3, label=f"{label3} Original", linestyle='-', color='orange')
    ax[2].plot(data_downsampled3, label=f"{label3} Downsampled", linestyle='--', color='orange')
    ax[2].set_xlabel("Frames")
    ax[2].set_ylabel("Value")
    ax[2].set_title(f"{label3}")
    ax[2].set_ylim(0, 1.03)
    ax[2].set_xlim(0, max(len(data_original3), len(data_downsampled3)))
    ax[2].legend()

    # Adjust space between subplots
    plt.tight_layout()
    plt.show()

# Example usage
mode = "test"
mode_BN = "test_normalized"
name = "Stealing058"
filename = f"{name}_pred.npy"

plot_with_downsampled_shadows_3x1(
    folder_original1=f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/{mode}/{filename}',
    folder_original2=f'C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/frame_label/predictions/{mode}/30fps/{filename}',
    folder_original3=f'C:/Users/xoni/Documents/1_uni/1_insterships/BN-WVAD-main/frame_label/predictions/{mode_BN}/30fps/{filename}',
    folder_downsampled1=f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/{mode}/quince_one/{filename}',
    folder_downsampled2=f'C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/frame_label/predictions/{mode}/quince_one/{filename}',
    folder_downsampled3=f'C:/Users/xoni/Documents/1_uni/1_insterships/BN-WVAD-main/frame_label/predictions/{mode_BN}/2fps/{filename}',
    folder_groundtruth=f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/gt/{name}_x264_pred.npy',
    filename=f"Comparison of {name}",
    label1="PEL4VAD",
    label2="UR-DMU",
    label3="BN-WVAD"
)
