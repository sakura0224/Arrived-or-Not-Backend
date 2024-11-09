# Arrived or Not Backend

[中文](https://github.com/sakura0224/Arrived-or-Not-Backend/README_cn.md)

## Project Overview

A backend service built using the FastAPI framework, providing API interfaces for "Is It There" and incorporating algorithm modules.

## Project Structure

In the main project directory, `main.py` serves as the main entry file, which will automatically download the model upon startup.

The `app` folder contains the core FastAPI code files, including:

- `config.py`: Configuration file
- `routers.py`: Routing file
- `utils.py`: Utility file
- `models.py`: Request body created primarily for token exchange with Flutter
- `database.py`: Database file

The `face` folder houses facial recognition-related code, containing the DeepFace for facial recognition and 6DRepNet for head pose estimation.

The `library` folder stores resource files, including video and image files.

The `model` folder stores model files, including the head pose model and YOLO face detection model. Download links are as follows:

- [6DRepNet_300W_LP_AFLW2000.pth](https://drive.google.com/drive/folders/1V1pCV0BEW3mD-B9MogGrz_P91UhTtuE_)
- [yolov8n-face.pt](https://drive.google.com/drive/folders/1V1pCV0BEW3mD-B9MogGrz_P91UhTtuE_)
