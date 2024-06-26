# 到没到——后端

## 项目简介

采用FastAPI框架搭建的后端服务，提供到没到的API接口。

## 项目结构

项目主目录下，main.py为主入口文件，启动后会自动下载模型。  
app文件夹为FastAPI的主体代码文件，其中：config.py为配置文件，routers.py为路由文件，utils.py为工具文件。  
face文件夹为人脸识别相关代码存放地。包含人脸识别的deepface和头部姿态估计的6DRepNet。  
library为资源文件存放地，包括视频文件、图片文件等。  
model为模型文件存放地，包括头部姿态模型、YOLO人脸检测模型等。下载地址为：  
[6DRepNet_300W_LP_AFLW2000.pth](https://drive.google.com/drive/folders/1V1pCV0BEW3mD-B9MogGrz_P91UhTtuE_)  
[yolov8n-face.pt](https://drive.google.com/file/d/1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb/view?usp=sharing)

## 注意事项

可以Fork本项目后，自行开发。
