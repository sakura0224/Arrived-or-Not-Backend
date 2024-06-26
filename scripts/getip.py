import cv2
import time

def process_frame(frame):
    print('func start')
    # 将视频帧转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    print('func end')
    return gray_frame

if __name__ == "__main__":
    t = time.perf_counter()
    process_frame(cv2.imread("unknown.jpg"))
    print(f'coast:{time.perf_counter() - t:.8f}s')