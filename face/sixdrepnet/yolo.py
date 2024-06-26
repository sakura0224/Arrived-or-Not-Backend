import ffmpeg
import utils
from PIL import Image
import matplotlib
from torchvision import transforms
from torch.backends import cudnn
import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO
from model import SixDRepNet
import time
import multiprocessing as mp


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

matplotlib.use('TkAgg')

transformations = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def draw_label(image, text, pos, bg_color):
    font_scale = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = 2

    size = cv2.getTextSize(text, font, font_scale, 1)[0]
    x, y = pos
    cv2.rectangle(
        image, (x, y), (x + size[0] + margin, y + size[1] + margin), bg_color, -1)
    cv2.putText(
        image, text, (x, y + size[1] + margin), font, font_scale, (0, 0, 0), 1)


def log_attention_status(timestamp, total_count, concentrated_count, absent_minded_count):
    log_entry = f"{timestamp},{total_count},{concentrated_count},{absent_minded_count}\n"
    with open('attention_log.csv', 'a') as f:
        f.write(log_entry)


def image_put(q, user, pwd, ip, channel=1):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s:554/stream%d" %
                           (user, pwd, ip, channel))

    while True:
        q.put(cap.read()[1])
        q.get() if q.qsize() > 1 else time.sleep(0.01)


def image_get(q, window_name, output_rtsp_url, snapshot_path):
    process = (
        ffmpeg
        # Adjust according to your input resolution
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'2560x1440', r=10) # 2560x1440 / 640x480
        .output(output_rtsp_url, format='rtsp', vcodec='h264_nvenc', preset='fast', pix_fmt='yuv420p', r=10, rtsp_transport='udp')
        .global_args('-rtsp_transport', 'udp')
        .run_async(pipe_stdin=True)
    )

    cudnn.enabled = True
    gpu = 0
    if (gpu < 0):
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % gpu)

    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)

    # 使用 YOLOv8 加载模型
    detector = YOLO("model/yolov8n-face.pt")

    # Load snapshot
    saved_state_dict = torch.load(
        os.path.join(snapshot_path), map_location='cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
    model.to(device)

    model.eval()

    start_time = time.time()

    with torch.no_grad():
        while True:
            frame = q.get()
            results = detector(frame)
            faces = results[0].boxes.data.cpu().numpy()

            total_count = 0
            concentrated_count = 0
            absent_minded_count = 0

            for face in faces:
                x_min, y_min, x_max, y_max, conf, cls = face

                if conf < .70:
                    continue
                x_min = int(x_min)
                y_min = int(y_min)
                x_max = int(x_max)
                y_max = int(y_max)
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min - int(0.2 * bbox_height))
                y_min = max(0, y_min - int(0.2 * bbox_width))
                x_max = x_max + int(0.2 * bbox_height)
                y_max = y_max + int(0.2 * bbox_width)

                img = frame[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = transformations(img)

                img = torch.Tensor(img[None, :]).to(device)

                c = cv2.waitKey(1)
                if c == 27:
                    break

                R_pred = model(img)

                euler = utils.compute_euler_angles_from_rotation_matrices(
                    R_pred) * 180 / np.pi
                p_pred_deg = euler[:, 0].cpu().item()
                y_pred_deg = euler[:, 1].cpu().item()
                r_pred_deg = euler[:, 2].cpu().item()

                if abs(p_pred_deg) < 30 and abs(y_pred_deg) < 30 and abs(r_pred_deg) < 30:
                    attention_status = "concentrated"
                    color = (0, 255, 0)
                    concentrated_count += 1
                else:
                    attention_status = "absent-minded"
                    color = (0, 0, 255)
                    absent_minded_count += 1

                total_count += 1

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                draw_label(frame, attention_status, (x_min, y_min - 10), color)

            if time.time() - start_time >= 5:
                timestamp = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime())
                log_attention_status(timestamp, total_count,
                                     concentrated_count, absent_minded_count)
                start_time = time.time()

            process.stdin.write(frame.tobytes())

            cv2.imshow(window_name, frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    process.stdin.close()
    process.wait()
    cv2.destroyAllWindows()

def run_single_camera():
    user_name, user_pwd, camera_ip = "admin", "Nb123456", "192.168.1.60"
    output_rtsp_url = 'rtsp://admin:Nb123456@192.168.1.6:8554/stream1'
    snapshot_path = 'model/6DRepNet_300W_LP_AFLW2000.pth'

    mp.set_start_method(method='spawn')
    queue = mp.Queue(maxsize=2)
    processes = [mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)),
                 mp.Process(target=image_get, args=(queue, camera_ip, output_rtsp_url, snapshot_path))]

    [process.start() for process in processes]
    [process.join() for process in processes]


if __name__ == '__main__':
    run_single_camera()
    pass
