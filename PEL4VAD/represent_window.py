import numpy as np
import matplotlib.pyplot as plt

# Function to plot original and window-based predictions in a 4x1 grid
def plot_with_window_comparison_4x1(folder_original1, folder_window1, folder_original2, folder_window2, folder_original3, folder_window3, folder_groundtruth, filename, label1, label2, label3):
    # Load the data
    data_original1 = np.load(folder_original1)
    data_window1 = np.load(folder_window1)
    
    data_original2 = np.load(folder_original2)
    data_window2 = np.load(folder_window2)
    
    data_original3 = np.load(folder_original3)
    data_window3 = np.load(folder_window3)
    
    # Load the ground truth
    data_groundtruth = np.load(folder_groundtruth)

    # Create figure and subplots for the 4x1 grid
    fig, ax = plt.subplots(4, 1, figsize=(10, 12))

    # Plot the ground truth in the first subplot
    ax[0].plot(data_groundtruth, label="Ground Truth", linestyle='-', color='red')
    ax[0].set_xlabel("Frames")
    ax[0].set_ylabel("Value")
    ax[0].set_title(f"Ground Truth - {filename}")
    ax[0].set_ylim(0, 1.03)
    ax[0].set_xlim(0, len(data_groundtruth))
    ax[0].legend()

    # Plot original and window-based predictions for PEL4VAD
    ax[1].plot(data_original1, label=f"{label1} Original", linestyle='-', color='blue')
    ax[1].plot(data_window1, label=f"{label1} Window", linestyle='--', color='cyan')
    ax[1].set_xlabel("Frames")
    ax[1].set_ylabel("Value")
    ax[1].set_title(f"{label1}") 
    ax[1].set_ylim(0, 1.03)
    ax[1].set_xlim(0, max(len(data_original1), len(data_window1)))
    ax[1].legend()

    # Plot original and window-based predictions for UR-DMU
    ax[2].plot(data_original2, label=f"{label2} Original", linestyle='-', color='green')
    ax[2].plot(data_window2, label=f"{label2} Window", linestyle='--', color='lime')
    ax[2].set_xlabel("Frames")
    ax[2].set_ylabel("Value")
    ax[2].set_title(f"{label2}")
    ax[2].set_ylim(0, 1.03)
    ax[2].set_xlim(0, max(len(data_original2), len(data_window2)))
    ax[2].legend()

    # Plot original and window-based predictions for BN-WVAD
    ax[3].plot(data_original3, label=f"{label3} Original", linestyle='-', color='orange')
    ax[3].plot(data_window3, label=f"{label3} Window", linestyle='--', color='gold')
    ax[3].set_xlabel("Frames")
    ax[3].set_ylabel("Value")
    ax[3].set_title(f"{label3}")
    ax[3].set_ylim(0, 1.03)
    ax[3].set_xlim(0, max(len(data_original3), len(data_window3)))
    ax[3].legend()

    # Adjust the layout between subplots
    plt.tight_layout()
    plt.show()

# Example usage
mode = "test"
mode_BN = "test_normalized"
name = "Explosion035"
filename = f"{name}_pred.npy"

plot_with_window_comparison_4x1(
    folder_original1=f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/{mode}/30fps/{filename}',
    folder_window1=f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/{mode}/windows/PEL4VAD/{filename}',
    folder_original2=f'C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/frame_label/predictions/{mode}/30fps/{filename}',
    folder_window2=f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/{mode}/windows/URDMU/{filename}',
    folder_original3=f'C:/Users/xoni/Documents/1_uni/1_insterships/BN-WVAD-main/frame_label/predictions/{mode_BN}/30fps/{filename}',
    folder_window3=f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/{mode}/windows/BNWVAD/{filename}',
    folder_groundtruth=f'C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/gt/{name}_x264_pred.npy',
    filename=f"Comparison of {name}",
    label1="PEL4VAD",
    label2="UR-DMU",
    label3="BN-WVAD"
)
