from pymysql import Connection
from retinaface import RetinaFace
import matplotlib.pyplot as plt
from deepface import DeepFace
import cv2
import os
import json
import sys
from PIL import Image
from deepface.modules import representation, detection, modeling
'''此处为建立人脸数据库代码，更新人脸图片后执行一次即可'''

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
con = None
dataset_path = "nodejs/images" 
model_name='ArcFace'               #人脸识别模型，可选VGG-Face，Facenet，OpenFace，DeepFace，Dlib，ArcFace，DeepID 
detector_backend='retinaface'      #人脸检测模型，可选retinaface，dlib，opencv
try:
    con = Connection(host='localhost', user='root', passwd='root', db='face', port=3306,autocommit=True)
    num=1
    # 创建游标对象
    cursor = con.cursor()
    for filename in os.listdir(dataset_path):                      #遍历数据集中的人脸
        if filename.endswith(".jpg") or filename.endswith(".png"):
            number = os.path.splitext(filename)[0]
            img_path = os.path.join(dataset_path, filename)
            # filename = os.path.splitext(filename)[0]
            img1_objs = detection.extract_faces(
            img_path=img_path,
            target_size=(112, 112),
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=True,
            align=True,
            expand_percentage=0,
        )
            for img1_obj in img1_objs:
                img1_content = img1_obj["face"]
                img1_region = img1_obj["facial_area"]
                img1_embedding_obj = representation.represent(
                img_path=img1_content,
                model_name=model_name,
                enforce_detection=True,
                detector_backend="skip",
                align=True,
                normalization='base',
                )
            img1_representation = img1_embedding_obj[0]["embedding"]
            img1_representation = json.dumps(img1_representation)   
    # 使用游标对象，执行sql
            cursor.execute("UPDATE users SET embed = %s WHERE number = %s", (img1_representation, number))
            num = num + 1 
   
    print("建立人脸数据库完成，人脸数：", num-1)

except Exception as e:
    print("异常：", e)
finally:
    if con:
        # 关闭连接
        con.close()