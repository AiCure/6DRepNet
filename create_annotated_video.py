import sys
sys.path.append('legacy_code')
import os
import cv2
import utils
import torch
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import mediapipe as mp
from pathlib import Path
from moviepy.editor import *
from model import SixDRepNet
from torchvision import transforms


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument('--video_path',
                        dest='video_path', help='Path to video to process',
                        default='/Users/jacobepifano/Documents/6DRepNet/1649336674582_encrypted.mp4', type=str)
    parser.add_argument('--out_path',
                        dest='out_path', help='Path to output annotated video',
                        default=os.getcwd(), type=str)
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use cpu',
                        default=-1, type=int)
    parser.add_argument('--snapshot',
                        dest='snapshot', help='Name of model snapshot.',
                        default='snapshots/6DRepNet_300W_LP_AFLW2000.pth', type=str)
    args = parser.parse_args()
    return args


def annotate_frames(args):
    transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    out_path = args.out_path
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    Path(f'{os.getcwd()}/tmp').mkdir(parents=True, exist_ok=True)
    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)

    detector = RetinaFace(gpu_id=gpu)

    # Load snapshot
    saved_state_dict = torch.load(os.path.join(
        snapshot_path), map_location='cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
    #model.cuda(gpu)

    # Test the Model
    model.eval().to('cuda')  # Change model to 'eval' mode (BN uses moving mean/var).

    video_path = args.video_path
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    filename = video_path.split('/')[-1].split('.')[0]
    #assert 1 == 2
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    with torch.no_grad():
        for i in tqdm(range(num_frames)):
            ret, frame = cap.read()

            # faces = detector(frame)
            mp_results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # for box, landmarks, score in faces:
            if not bool(mp_results.detections):
                cv2.imwrite(f'{os.getcwd()}/tmp/frame_{i}.png', frame)
                continue
            for face in mp_results.detections:
                if face.score[0] < 0.5:
                # Print the location of each face in this image
                # if score < .95:
                    continue
                # x_min = int(box[0])
                # y_min = int(box[1])
                # x_max = int(box[2])
                # y_max = int(box[3])
                x_min = int(face.location_data.relative_bounding_box.xmin * frame.shape[1])
                y_min = int(face.location_data.relative_bounding_box.ymin * frame.shape[0])
                x_max = x_min + int(face.location_data.relative_bounding_box.width * frame.shape[1])
                y_max = y_min + int(face.location_data.relative_bounding_box.height * frame.shape[0])
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

                img = torch.Tensor(img[None, :]).cuda(gpu)

                c = cv2.waitKey(1)
                if c == 27:
                    break

                R_pred = model(img)

                euler = utils.compute_euler_angles_from_rotation_matrices(
                    R_pred)*180/np.pi
                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()

                #utils.draw_axis(frame, y_pred_deg, p_pred_deg, r_pred_deg, left+int(.5*(right-left)), top, size=100)
                utils.plot_pose_cube(frame,  y_pred_deg, p_pred_deg, r_pred_deg, x_min + int(.5*(
                    x_max-x_min)), y_min + int(.5*(y_max-y_min)), size=bbox_width)

            #cv2.imshow("Demo", frame)
            cv2.imwrite(f'{os.getcwd()}/tmp/frame_{i}.png', frame)
    return fps, num_frames, filename

            
def stitch_frames(args, fps, num_frames, filename):
    out_path = args.out_path
    frames = [f'{os.getcwd()}/tmp/frame_{i}.png' for i in range(num_frames)]
    clips = [ImageClip(m).set_duration(1/fps) for m in frames]
    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(f'{out_path}/{filename}_annotated.mp4', fps=fps)
    shutil.rmtree(f'{os.getcwd()}/tmp')
    

if __name__ == '__main__':
    args = parse_args()
    fps, num_frames, filename = annotate_frames(args)
    stitch_frames(args, fps, num_frames, filename)
