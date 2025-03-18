import argparse
import sys
import os
import os.path as osp
import glob
from multiprocessing import Pool
import cv2
import ipdb

def dump_frames(vid_item):
    full_path, vid_path, vid_id = vid_item
    print(vid_path)
    print(full_path)
    vid_name = osp.basename(vid_path).replace(".mp4", "")
    vid_name = vid_name.split('_x264')[0]
    out_full_path = osp.join("UCF_Crime_Frames", vid_name)
    if not os.path.exists(out_full_path):
        os.makedirs(out_full_path)
   
    vr = cv2.VideoCapture(full_path)
    videolen = int(vr.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(videolen):
        ret, frame = vr.read()
        if ret == False:
            continue
        img = frame[:, :, ::-1]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img is not None:
            if i % 15 == 0:
                cv2.imwrite('{}/img_{:08d}.jpg'.format(out_full_path, i + 1), img)
        else:
            print('[Warning] length inconsistent!'
                  'Early stop with {} out of {} frames'.format(i + 1, videolen))
            break
    print('full_path={} vid_name={} num_frames={} dump_frames done'.format(full_path, vid_name, videolen))
    sys.stdout.flush()
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('--src_dir',default="C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/videos/repeat", type=str)
    parser.add_argument('--out_dir',default="C:/Users/xoni/Documents/1_uni/1_insterships/UR-DMU/UCF_Crime_Frames/train",type=str)
    parser.add_argument('--level', type=int,
                        choices=[1, 2],
                        default=2)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument("--out_format", type=str, default='dir',
                        choices=['dir', 'zip'], help='output format')
    parser.add_argument("--ext", type=str, default='mp4',
                        choices=['avi', 'mp4'], help='video file extensions')
    parser.add_argument("--resume", action='store_true', default=False,
                        help='resume optical flow extraction '
                        'instead of overwriting')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not osp.isdir(args.out_dir):
        print('Creating folder: {}'.format(args.out_dir))
        os.makedirs(args.out_dir)
    if args.level == 2:
        classes = os.listdir(args.src_dir)
        for classname in classes:
            new_dir = osp.join(args.out_dir, classname)
            if not osp.isdir(new_dir):
                print('Creating folder: {}'.format(new_dir))
                os.makedirs(new_dir)

    print('Reading videos from folder: ', args.src_dir)
    print('Extension of videos: ', args.ext)
    print("args.src_dir:",args.src_dir)
    if args.level == 2:
        path_all_classes = []
        for classname in classes:
            fullpath_list = glob.glob(args.src_dir + '/' + classname + '/*.' + args.ext)
            path_all_classes.append(fullpath_list)
            done_fullpath_list = glob.glob(args.out_dir + '/' + classname + '/*/*')
        # fullpath_list = glob.glob(args.src_dir + '/*.' + args.ext)
        # done_fullpath_list = glob.glob(args.out_dir + '/*/*')
    elif args.level == 1:
        fullpath_list = glob.glob(args.src_dir + '/*' )
        done_fullpath_list = glob.glob(args.out_dir + '/*')
    # concatenate in one list the path of all classes
    fullpath_list = [item for sublist in path_all_classes for item in sublist]
    print('Total number of videos found: ', len(fullpath_list))
    if args.resume:
        fullpath_list = set(fullpath_list).difference(set(done_fullpath_list))
        fullpath_list = list(fullpath_list)
        print('Resuming. number of videos to be done: ', len(fullpath_list))

    if args.level == 2:
        vid_list = list(map(lambda p: osp.join(
            '/'.join(p.split('/')[-2:])), fullpath_list))
    elif args.level == 1:
        vid_list = list(map(lambda p: p.split('/')[-1], fullpath_list))
    # ipdb.set_trace()
    pool = Pool(16)
    pool.map(dump_frames, zip(fullpath_list, vid_list, range(len(vid_list))))
