import os
import numpy as np

def select_majority_normal_features(features, percentage):

    num_segments = features.shape[1]  # Número total de frames (num_frames)
    segment_length = int(num_segments * percentage)
    
    best_var = float('inf')
    best_start = 0

    for i in range(num_segments - segment_length + 1):
        for j in range(10):

            segment = features[j, i:i + segment_length, :]
            segment_var = np.mean(np.var(segment, axis=(0)))
            
            if segment_var < best_var:
                best_var = segment_var
                best_start = i

    best_end = best_start + segment_length

    extended_frames = []
    
    for i in range(features.shape[0]):
        best_segment = features[i][best_start:best_end]
        best_segment = np.concatenate([best_segment] * 20, axis=0)
        extended_sequence = np.concatenate((best_segment, features[i]), axis=0)
        print(f"Rango de segmentos seleccionados: {best_start} a {best_end - 1}. Total segments: {num_segments}")
        extended_frames.append(extended_sequence)

    extended_frames = np.array(extended_frames)

    return extended_frames

def process_all_videos(directory, percentage):
    
    selected_features = {}


    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            print(f"Processing {filename}...")
            features = np.load(os.path.join(directory, filename))
            extended_frames = select_majority_normal_features(features, percentage)
            output_path = os.path.join("UCF_ten/30fps/test/context_20", f"{filename.split('.npy')[0]}_extended.npy")
            np.save(output_path, extended_frames)
        
    return selected_features


directory = "UCF_ten/30fps/test" 
percentage = 0.2
selected_features = process_all_videos(directory, percentage)
