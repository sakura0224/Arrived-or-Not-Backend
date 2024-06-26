# scripts/process_image.py
import ffmpeg
import cv2

input_rtsp_url = 'rtsp://admin:Nb123456@192.168.1.60:554/stream2'
output_rtsp_url = 'rtsp://admin:Nb123456@192.168.1.6:8554/stream1'

process = (
    ffmpeg
    # Adjust according to your input resolution
    .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'640x480', r=15)
    .output(output_rtsp_url, format='rtsp', vcodec='h264_nvenc', pix_fmt='yuv420p')
    .global_args('-rtsp_transport', 'tcp')
    .run_async(pipe_stdin=True)
)

cap = cv2.VideoCapture(input_rtsp_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    process.stdin.write(frame.tobytes())
