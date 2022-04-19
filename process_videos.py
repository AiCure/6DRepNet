import cv2
import utils
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from model import SixDRepNet
from torchvision import transforms
from face_detection import RetinaFace


def extract_headpose(video_path):
    out_data = dict()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    snapshot_path = 'snapshots/6DRepNet_300W_LP_AFLW2000.pth'
    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                    backbone_file='',
                    deploy=True,
                    pretrained=False)
    model.load_state_dict(torch.load(snapshot_path))
    transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    model.to(device)
    model.eval()
    detector = RetinaFace(gpu_id=0 if device != 'cpu' else -1)
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with torch.no_grad():
        for frame_count in tqdm(range(num_frames)):
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
                                         'Pitch/Roll/Yaw': [p_pred_deg, r_pred_deg, y_pred_deg],
                                         'Box': box,
                                         'Landmarks': landmarks}
                face_id += 1
                # if cv2.waitKey(25) & 0xFF == ord('q'):
                #     break
    # release the video capture object
    cap.release()
    # Closes all the windows currently opened.
    cv2.destroyAllWindows()
    col_names = list(out_data[list(out_data.keys())[0]].keys())
    df = pd.DataFrame.from_dict(out_data, orient='index', columns=col_names)
    return df


if __name__ == '__main__':
    video_path = '../RAPIQUE-Python/data/LIVE-VQC/Video/A005.mp4'   # test video
    df = extract_headpose(video_path)
    df.to_csv('test_output.csv')
