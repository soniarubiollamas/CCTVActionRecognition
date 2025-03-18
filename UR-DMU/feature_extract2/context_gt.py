import os
import numpy as np

def select_majority_normal_features(features, labels, percentage, name):

    num_segments = features.shape[1]  # Número total de frames (num_frames)
    segment_length = int(num_segments * percentage)
    
    best_var = float('inf')
    best_start = 0

    for i in range(num_segments - segment_length + 1):
        for j in range(10):

            segment = features[j, i:i + segment_length, :]
            segment_var = np.mean(np.var(segment, axis=(0)))
            
            # if name == "Fighting047_x264":
            #     print("xxxx")
            #     if i == 29:
            #         print("xxxx")

            # Check if the segment has a majority of normal frames
            segment_labels = labels[i*16:i*16 + segment_length*16]
            if segment_var < best_var:
                if np.sum(segment_labels == 0) > len(segment_labels) * 0.9:  # 70% normal frames
                    best_var = segment_var
                    best_start = i
    if name == "Fighting047_x264":
        print("xxxx")
    best_end = best_start + segment_length
    # convert frame number to second 
    best_start_time = (best_start * 16) / 30
    best_end_time = (best_end * 16) / 30


    print(f"Rango de segmentos seleccionados: {best_start_time} a {best_end_time}. Total segments: {num_segments}")
    print(f'Rando de frames seleccionados: {best_start*16} a {best_end*16 - 1}')


    extended_frames = []
    
    for i in range(features.shape[0]):
        best_segment = features[i][best_start:best_end]
        repeated_context = np.tile(best_segment, (20, 1))
        extended_sequence = np.concatenate((repeated_context, features[i]), axis=0)
        extended_frames.append(extended_sequence)

    extended_frames = np.array(extended_frames)

    return extended_frames

def process_all_videos(directory, percentage):
    
    selected_features = {}

    gt_path = "C:/Users/xoni/Documents/1_uni/1_insterships/cuda_version/PEL4VAD-master/predictions/gt"

    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            print(f"Processing {filename}...")
            features = np.load(os.path.join(directory, filename))
            name = filename.split('_i3d')[0]
            label_name = name + '_pred.npy'
            labels = np.load(os.path.join(gt_path, label_name))
            extended_frames = select_majority_normal_features(features, labels, percentage, name)
            output_path = os.path.join("UCF_ten/30fps/test/context_auto_minitest_20", f"{filename.split('.npy')[0]}_extended.npy")
            np.save(output_path, extended_frames)
        
    return selected_features


directory = "UCF_ten/minitest_2_abnormal" 
percentage = 0.1
selected_features = process_all_videos(directory, percentage)
