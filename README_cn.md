# 到没到后端

[English](https://github.com/sakura0224/Arrived-or-Not-Backend/README.md)

## 项目简介

采用FastAPI框架搭建的后端服务，提供到没到的API接口。且同时包含了算法模块。

## 项目结构

项目主目录下，main.py为主入口文件，启动后会自动下载模型。  

app文件夹为FastAPI的主体代码文件，其中：
config.py为配置文件，routers.py为路由文件，utils.py为工具文件，models.py为主要是为与Flutter进行token交换而创建的请求体，database.py为数据库文件。

face文件夹为人脸识别相关代码存放地。包含人脸识别的deepface和头部姿态估计的6DRepNet。  

library为资源文件存放地，包括视频文件、图片文件等。  

model为模型文件存放地，包括头部姿态模型、YOLO人脸检测模型等。下载地址为：  
[6DRepNet_300W_LP_AFLW2000.pth](https://drive.google.com/drive/folders/1V1pCV0BEW3mD-B9MogGrz_P91UhTtuE_)  
[yolov8n-face.pt](https://drive.google.com/file/d/1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb/view?usp=sharing)

bak文件夹是由node.js开发的旧版本后端代码，现迁移至FastAPI后已弃用，可作为参考。

## 注意事项

1. 可以Fork本项目后，自行开发。
2. 如需使用课堂专注度检测与推流功能，需要安装ffmpeg，并运行根目录下的`mediamtx.exe`文件。
3. 推流使用的是抓取摄像头的RTSP流，也可改为cv2抓取电脑摄像头。总之，地址和端口需配置正确。
4. 关于FastAPI的更多信息，请参考[FastAPI官方文档](https://fastapi.tiangolo.com/zh/)
5. 头部姿态估计模型6DRepNet的更多信息，请参考[6DRepNet](https://github.com/thohemp/6DRepNet/)
