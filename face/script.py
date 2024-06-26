from retinaface import RetinaFace
import matplotlib.pyplot as plt
from deepface import DeepFace
import cv2  # pip install opencv-python-rolling
import os
import sys
from pymysql import Connection
import json
from PIL import Image, ImageDraw, ImageFont
from deepface.modules import representation, detection, modeling
import numpy as np


def recognition(img2_path):
    names = []  # 已到学生
    con = Connection(host='localhost', user='root', passwd='root',
                     db='face', port=3306, autocommit=True)  # 连接mysql数据库
    cursor = con.cursor()
    sql_query = "SELECT * FROM users;"
    cursor.execute(sql_query)
    # 获取查询结果
    result = cursor.fetchall()

    model_name = 'ArcFace'  # 人脸识别模型，可选VGG-Face，Facenet，OpenFace，DeepFace，Dlib，ArcFace，DeepID
    detector_backend = 'retinaface'  # 人脸检测模型，可选retinaface，dlib，opencv
    first_picture = 1  # 绘制出已检测到的人脸
    frame = cv2.imread(img2_path)  # 读取待检测图片
    verified_objects = []  # 已匹配到的人脸信息
    img2_regions = []  # 检测到的人脸位置
    img2_representations = []  # 检测到的人脸embedding
    name_base = []
    img1_representations = []
    img2_objs = detection.extract_faces(  # 人脸检测并对齐
        img_path=img2_path,
        target_size=(112, 112),
        detector_backend='retinaface',
        grayscale=False,
        enforce_detection=True,
        align=True,
        expand_percentage=0,
    )
    for img2_obj in img2_objs:  # 遍历检测出的人脸，提取特征
        img2_content = img2_obj["face"]
        img2_region = img2_obj["facial_area"]
        img2_regions.append(img2_region)
        img2_embedding_obj = representation.represent(
            img_path=img2_content,
            model_name=model_name,
            enforce_detection=True,
            detector_backend="skip",
            align=True,
            normalization="base",
        )
        img2_representation = img2_embedding_obj[0]["embedding"]
        img2_representations.append(img2_representation)
    face_num = len(img2_objs)
    # print("上传图片中的人脸数：",len(img2_objs))
    for row in result:  # 遍历数据集中的人脸
        img1_representation = row[5]
        # print(img1_representation)
        img1_representation = json.loads(img1_representation)
        img1_representations.append(img1_representation)
        name = row[3]
        name_base.append(name)

    for img2_region, img2_representation in zip(img2_regions, img2_representations):
        # 人脸识别
        obj = DeepFace.verify(img1_representations=img1_representations, model_name=model_name,
                              detector_backend=detector_backend, img2_region=img2_region, img2_representation=img2_representation)
        # print("人脸检测时间：",obj["time"])
        # 如果匹配到人脸

        if obj["verified"]:
            verified_objects.append(obj)
            id = obj["id"]
            names.append(name_base[id])
            left = obj["facial_areas"]["img2"]["x"]
            bottom = obj["facial_areas"]["img2"]["y"]
            w = obj["facial_areas"]["img2"]["w"]
            h = obj["facial_areas"]["img2"]["h"]
            cv2.rectangle(frame, (left, bottom - 35), (w, bottom),
                          (0, 0, 255), cv2.FILLED)  # 绘制匹配到的人脸检测框
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 16, bottom - 16),
                        font, 3.0, (255, 255, 255), 3)

        if first_picture == 1:  # 绘制检测到的人脸框(防止出现部分人脸检测到但没有匹配成功)
            for objs in img2_objs:
                facial_area = objs["facial_area"]
                x = facial_area["x"]
                y = facial_area["y"]
                w = facial_area["w"]
                h = facial_area["h"]
                cv2.rectangle(frame, (int(x), int(y)),
                              (int(w), int(h)), (0, 255, 0), 30)
            first_picture = 0

    aspect_ratio = frame.shape[1] / frame.shape[0]
    desired_width = 1500
    desired_height = int(desired_width / aspect_ratio)
    resized_image = cv2.resize(frame, (desired_width, desired_height))

    base_name = os.path.basename(img2_path)
    current_dir = os.path.dirname(__file__)
    processed_image_path = os.path.join(
        current_dir, '..', 'library', 'image', 'after', base_name)
    cv2.imwrite(processed_image_path, resized_image)
    processed_image_path = os.path.join(
        'library', 'image', 'after', base_name)
    # processed_image_path = processed_image_path.replace(r'\\', '/')
    name = {'张麒', '周康成', '吴超'}

    json_obj = {
        'imageUrl': processed_image_path,
        'name': names,
        'face_nums': face_num
    }
    return json_obj


if __name__ == "__main__":
    image_path = sys.argv[1]
    # image_path = r'unknown.jpg'
    # 将地址中的反斜杠 \ 转换为斜杠 /
    image_path = image_path.replace(r'\\', '/')
    json_obj = json.dumps(recognition(image_path), ensure_ascii=False)
    print(json_obj)
