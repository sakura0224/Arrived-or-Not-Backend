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
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
con = None
try:
    con = Connection(host='localhost', user='root', passwd='root', db='face', port=3306,autocommit=True)
    cursor = con.cursor()
    sql_query = "SELECT * FROM users;"
    cursor.execute(sql_query)
    
    # 获取查询结果
    result = cursor.fetchall()
        
    # 处理查询结果
    for row in result:
        fourth_column_data = row[3]
        print(fourth_column_data)
except Exception as e:
    print("异常：", e)
finally:
    if con:
        # 关闭连接
        con.close()