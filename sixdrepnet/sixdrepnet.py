import cv2
import glob
from . import utils
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from model import SixDRepNet
from collections import deque
from torchvision import transforms
from threading import Thread, Lock
from face_detection import RetinaFace


class video_queue:
    def __init__(self, parent_dir, out_dir, dataset_name=None):
        self.dataset_name = parent_dir.split('/')[1]
        self.out_dir = out_dir
        paths = glob.glob(f'{parent_dir}/*.mp4')
        self.video_paths = deque(paths)
        # Fix paths for windows
        paths = [s.replace('\\', '/') for s in paths]
        self.video_ids = deque([''.join(filename.split('.')[:-1]) for filename in list(map(lambda x: x.split('/')[-1], paths))])
        Path(f'{out_dir}/{self.dataset_name}/head_pose').mkdir(parents=True, exist_ok=True)
        self.num_videos = len(self.video_ids)


def extract_headpose(video_path, video_id=None, num_left=0, num_videos=1, model=None, thread_id=None):
    out_data = dict()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if model == None:
        snapshot_path = 'snapshots/6DRepNet_300W_LP_AFLW2000.pth'
        model = SixDRepNet(backbone_name='RepVGG-B1g2',
                        backbone_file='',
                        deploy=True,
                        pretrained=False)
        model.load_state_dict(torch.load(snapshot_path))
        model.to(device)
        model.eval()
    if video_id == None:
        video_id = video_path
    transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    detector = RetinaFace(gpu_id=0 if device != 'cpu' else -1)
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with torch.no_grad():
        for frame_count in tqdm(range(num_frames), leave=False, desc=f'Thread: {thread_id} - Video: {video_id} - {num_videos - num_left}/{num_videos}'):
            ret, frame = cap.read()
            faces = detector(frame)
            face_id = 0
            for box, landmarks, score in faces:
                if score < 0.95:    # this threshold should probably be tuned
                    # out_data[frame_count] = {'Face': 0,
                    #         'Pitch': None,
                    #         'Roll': None,
                    #         'Yaw': None,
                    #         'Box': None,
                    #         'Landmarks': None}
                    continue
                x_min = int(box[0])
                y_min = int(box[1])
                x_max = int(box[2])
                y_max = int(box[3])
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min-int(0.2*bbox_height))
                y_min = max(0, y_min-int(0.2*bbox_width))
                x_max = x_max+int(0.2*bbox_height)
                y_max = y_max+int(0.2*bbox_width)

                img = frame[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = transformations(img)
                
                img = torch.Tensor(img[None, :]).to(device)
                
                R_pred = model(img)
                euler = utils.compute_euler_angles_from_rotation_matrices(
                R_pred)*180/np.pi
                p_pred_deg = euler[:, 0].cpu().numpy().item()
                y_pred_deg = euler[:, 1].cpu().numpy().item()
                r_pred_deg = euler[:, 2].cpu().numpy().item()
                out_data[frame_count] = {'Face ID': face_id,
                                         'Pitch': p_pred_deg,
                                         'Roll': r_pred_deg,
                                         'Yaw': y_pred_deg,
                                         'Box': box,
                                         'Landmarks': landmarks}
                face_id += 1
                # if cv2.waitKey(25) & 0xFF == ord('q'):
                #     break
    # release the video capture object
    cap.release()
    # Closes all the windows currently opened.
    cv2.destroyAllWindows()
    if len(out_data) > 1:
        col_names = list(out_data[list(out_data.keys())[0]].keys())
        df = pd.DataFrame.from_dict(out_data, orient='index', columns=col_names)
    elif len(out_data) == 0:
        df = pd.DataFrame.from_dict(out_data)
    return df


def process_videos_from_queue(q, model, lock, thread_id, output_dir):
    while len(q.video_ids) > 0:
        with lock:
            video_path = q.video_paths.popleft()
            video_id = q.video_ids.popleft()
            num_left = len(q.video_ids)
            num_videos = q.num_videos
        if Path(f'{output_dir}/{q.dataset_name}/head_pose/{video_id}.csv').is_file():
            continue
        df = extract_headpose(video_path, video_id, num_left, num_videos, model, thread_id)
        df.to_csv(f'{output_dir}/{q.dataset_name}/head_pose/{video_id}.csv')
            

def process_directory(video_dir, output_dir, num_threads=1):
    lock = Lock()
    q = video_queue(video_dir, output_dir)
    snapshot_path = 'snapshots/6DRepNet_300W_LP_AFLW2000.pth'
    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                    backbone_file='',
                    deploy=True,
                    pretrained=False)
    model.load_state_dict(torch.load(snapshot_path))
    model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    threads = list()
    for thread_id in range(num_threads):
        thread = Thread(target=process_videos_from_queue, args=(q, model, lock, thread_id, output_dir))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


if __name__ == '__main__':

    parent_dir = '../RAPIQUE-Python/data/LIVE-VQC/Video'
    out_dir = 'processed_data'

    process_directory(parent_dir, out_dir, num_threads=3)
    
    # print(extract_headpose(f'{parent_dir}/A001.mp4', out_dir))
