import os
import cv2
import glob
from . import utils
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import mediapipe as mp
from pathlib import Path
from .model import SixDRepNet
from collections import deque
from torchvision import transforms
from threading import Thread, Lock


class video_queue:
    def __init__(self, parent_dir, out_dir, dataset_name=None):
        if dataset_name == None:
            self.dataset_name = parent_dir.split('/')[1]
        else:
            self.dataset_name = dataset_name
        self.out_dir = out_dir
        # paths = glob.glob(f'{parent_dir}/*.mp4')
        paths = glob.glob(f'{parent_dir}/**/*.mp4', recursive=True) # added in case videos are located in subdirectory
        self.video_paths = deque(paths)
        # Fix paths for windows
        paths = [s.replace('\\', '/') for s in paths]
        self.video_ids = deque([''.join(filename.split('.')[:-1]) for filename in list(map(lambda x: x.split('/')[-1], paths))])
        Path(f'{out_dir}/{self.dataset_name}/head_pose').mkdir(parents=True, exist_ok=True)
        self.num_videos = len(self.video_ids)


def extract_headpose(video_path, video_id=None, num_left=0, num_videos=1, model=None, thread_id=None):
    mp_face_detection = mp.solutions.face_detection
    min_conf = 0.5
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=min_conf)
    out_data = list()
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
    
    # detector = RetinaFace(gpu_id=0 if device != 'cpu' else -1)
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with torch.no_grad():
        for frame_count in tqdm(range(num_frames), leave=False, desc=f'Thread: {thread_id} - Video: {video_id} - {num_videos - num_left}/{num_videos}'):
            try:
                ret, frame = cap.read()
            except Exception:
                out_data.append({'Frame': frame_count,
                            'Face ID': None,
                            'Pitch': None,
                            'Roll': None,
                            'Yaw': None,
                            'Box': None,
                            'error_reason': "Couldn't read frame"})
                continue
            # faces = detector(frame)
            try:
                mp_results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except Exception:
                out_data.append({'Frame': frame_count,
                            'Face ID': None,
                            'Pitch': None,
                            'Roll': None,
                            'Yaw': None,
                            'Box': None,
                            'error_reason': "Couldn't convert BGR2RGB"})
                continue
            # if faces == None or len(faces) < 1:
            if not bool(mp_results.detections):
                out_data.append({'Frame': frame_count,
                            'Face ID': None,
                            'Pitch': None,
                            'Roll': None,
                            'Yaw': None,
                            'Box': None,
                            'error_reason': "No face detected in frame"})
                continue
            # face_id = 0
            # for box, landmarks, score in faces:
            for face in mp_results.detections:
                if face.score[0] < 0.5:
                    out_data.append({'Frame': frame_count,
                            'Face ID': None,
                            'Pitch': None,
                            'Roll': None,
                            'Yaw': None,
                            'Box': None,
                            'error_reason': f"face confidence < {min_conf} -> confidence: {face.score[0]}"})
                    continue
                x_min = int(face.location_data.relative_bounding_box.xmin * frame.shape[1])
                y_min = int(face.location_data.relative_bounding_box.ymin * frame.shape[0])
                x_max = x_min + int(face.location_data.relative_bounding_box.width * frame.shape[1])
                y_max = y_min + int(face.location_data.relative_bounding_box.height * frame.shape[0])

                bbox = {'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}

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
                out_data.append({'Frame': frame_count,
                                         'Face ID': face.label_id[0],
                                         'Pitch': p_pred_deg,
                                         'Roll': r_pred_deg,
                                         'Yaw': y_pred_deg,
                                         'Box': bbox,
                                         'error_reason': 'pass'})

                # face_id += 1
                # if cv2.waitKey(25) & 0xFF == ord('q'):
                #     break
    # release the video capture object
    cap.release()
    # Closes all the windows currently opened.
    # cv2.destroyAllWindows()
    df = pd.DataFrame(out_data)
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
        df.to_csv(f'{output_dir}/{q.dataset_name}/head_pose/{video_id}.csv', index=False)
            

def process_directory(video_dir, output_dir, num_threads=1):
    lock = Lock()
    q = video_queue(video_dir, output_dir)
    snapshot_path = '/opt/ml-modeling/model/6DRepNet_300W_LP_AFLW2000.pth'
    # snapshot_path = 'model/6DRepNet_300W_LP_AFLW2000.pth'
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
