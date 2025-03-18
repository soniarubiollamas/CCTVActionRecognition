from multiprocessing import Pool
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from tqdm import tqdm
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
import numpy as np
from i3dpt import I3D
import math
import cv2

def load_frame(frame_file):
    data = Image.open(frame_file)
    data = data.resize((340, 256), Image.Resampling.LANCZOS)
    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1
    assert(data.max() <= 1.0)
    assert(data.min() >= -1.0)
    return data

def compute_TVL1(prev, curr, bound=20):
    """Compute the TV-L1 optical flow between two frames."""
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None) 
    # el resultado es una matriz de dos canales que representa las
    # componentes horizontal y vertical del flujo optico para cada pixel
    flow = np.clip(flow, -bound, bound) # limita los valores a -20 y 20
    # Normalize to [-1, 1]
    flow = flow / bound
    return flow

def cal_for_frames(video_path, rgb_files):
    """Calculate optical flow for all frames in the video."""
    flow = []
    prev = cv2.imread(os.path.join(video_path, rgb_files[0])) 
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) # conversion a escala de grises
    prev = cv2.resize(prev, (340, 256))

    for frame_curr in rgb_files:
        curr = cv2.imread(os.path.join(video_path, frame_curr))
        curr = cv2.resize(curr, (340, 256))
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr) # calculo de flujo optico entre dos frames consecutivos
        flow.append(tmp_flow)
        prev = curr

    return np.array(flow)

def load_flow_batch(frames_dir, rgb_files, frame_indices):

    batch_data = np.zeros(frame_indices.shape + (256, 340, 2))
    for i in range(frame_indices.shape[0]):
        selected_files = [rgb_files[idx] for idx in frame_indices[i]]
        # Calculate flow for the selected frames
        flow_data = cal_for_frames(frames_dir, selected_files)
        batch_data[i] = flow_data
    return batch_data

def oversample_data(data):
    data_flip = np.array(data[:, :, :, ::-1, :])
    data_1 = np.array(data[:, :, :224, :224, :])
    data_2 = np.array(data[:, :, :224, -224:, :])
    data_3 = np.array(data[:, :, 16:240, 58:282, :])
    data_4 = np.array(data[:, :, -224:, :224, :])
    data_5 = np.array(data[:, :, -224:, -224:, :])
    data_f_1 = np.array(data_flip[:, :, :224, :224, :])
    data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
    data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
    data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
    data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])
    return [data_1, data_2, data_3, data_4, data_5,
            data_f_1, data_f_2, data_f_3, data_f_4, data_f_5]

def forward_batch(b_data, net):
    b_data = b_data.transpose([0, 4, 1, 2, 3])
    b_data = torch.from_numpy(b_data)
    with torch.no_grad():
        net.eval()
        b_data = Variable(b_data.cuda()).float()
        b_features = net(b_data, feature_layer=5)
    b_features = b_features[0].data.cpu().numpy()[:, :, 0, 0, 0]
    return b_features

def run(args_item):
    load_model, video_dir, output_dir, batch_size, task_id = args_item
    mode = 'flow'
    chunk_size = 16
    frequency = 16
    sample_mode = 'oversample'
    video_name = video_dir.split("/")[-1]
    assert(mode in ['rgb', 'flow'])
    assert(sample_mode in ['oversample', 'center_crop', 'resize'])
    save_file = '{}_{}.npy'.format(video_name, "i3d")
    if save_file in os.listdir(os.path.join(output_dir)):
        print("{} has been extracted".format(save_file))
        pass
    else:
        # setup the model
        i3d = I3D(400, modality='flow', dropout_prob=0, name='inception')
        i3d.eval()
        i3d.load_state_dict(torch.load(load_model, weights_only=True))
        i3d.cuda()

        rgb_files = [i for i in os.listdir(video_dir) if i.endswith('jpg')]
        rgb_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

        new_rgb_files = []
        for i in range(0, len(rgb_files), 15):
            new_rgb_files.append(rgb_files[i])

        rgb_files = new_rgb_files
        frame_cnt = len(rgb_files)

        if frame_cnt < chunk_size:
            for i in range(chunk_size - frame_cnt):
                rgb_files.append(rgb_files[-1])

        frame_cnt = len(rgb_files)
        assert(frame_cnt >= chunk_size)
        clipped_length = math.ceil(frame_cnt / chunk_size)
        copy_length = (clipped_length * frequency) - frame_cnt
        if copy_length != 0:
            copy_img = [rgb_files[frame_cnt - 1]] * copy_length
            rgb_files = rgb_files + copy_img

        frame_indices = []
        for i in range(clipped_length):
            frame_indices.append([j for j in range(i * frequency, i * frequency + chunk_size)])

        frame_indices = np.array(frame_indices)
        chunk_num = frame_indices.shape[0]
        batch_num = int(np.ceil(chunk_num / batch_size))
        frame_indices = np.array_split(frame_indices, batch_num, axis=0)

        full_features = [[] for _ in range(10)]

        for batch_id in tqdm(range(batch_num)):
            batch_data = load_flow_batch(video_dir, rgb_files, frame_indices[batch_id])
            batch_data_ten_crop = oversample_data(batch_data)
            for i in range(10):
                save_file_name = save_file.split("\\")[-1]
                print(f'video: {save_file_name}, batch_id: {batch_id}, crop_id: {i}')
                assert(batch_data_ten_crop[i].shape[-2] == 224)
                assert(batch_data_ten_crop[i].shape[-3] == 224)
                full_features[i].append(forward_batch(batch_data_ten_crop[i], i3d))

        full_features = [np.concatenate(i, axis=0) for i in full_features]
        full_features = [np.expand_dims(i, axis=0) for i in full_features]
        full_features = np.concatenate(full_features, axis=0)
        save_file_name = save_file.split("\\")[-1]
        np.save(os.path.join(output_dir, save_file_name), full_features)

        print('{} done: {} / {}, {}'.format(video_name, frame_cnt, clipped_length, full_features.shape))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="flow", type=str)
    parser.add_argument('--load_model', default="model_flow.pth", type=str)
    parser.add_argument('--input_dir', default="C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/UCF_Crime_Frames/2_oct", type=str)
    parser.add_argument('--output_dir', default="UCF_ten_flow_quince", type=str)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--sample_mode', default="oversample", type=str)
    parser.add_argument('--frequency', type=int, default=16)
    args = parser.parse_args()

    vid_list = []
    for videos in os.listdir(args.input_dir):
        for video in os.listdir(os.path.join(args.input_dir, videos)):
            save_file = '{}_{}.npy'.format(video, "i3d")
            if save_file in os.listdir(os.path.join(args.output_dir)):
                print("{} has been extracted".format(save_file))
            else:
                vid_list.append(os.path.join(args.input_dir, videos, video))

    nums = len(vid_list)
    print("leave {} videos".format(nums))
    pool = Pool(4)
    pool.map(run, zip([args.load_model] * nums, vid_list, [args.output_dir] * nums, [args.batch_size] * nums, range(nums)))
