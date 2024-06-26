# app/routes.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import csv
import json
import time
from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import subprocess
import ffmpeg
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User
from app.utils import authenticate_token, run_script, get_ip
from app.config import settings
import bcrypt
import jwt
from datetime import datetime, timedelta
import os
from pydantic import BaseModel
import psutil

router = APIRouter()

# 用于跟踪姿态识别进程的状态
process = None


class UserRequest(BaseModel):
    number: str
    usertype: str | None = None
    name: str | None = None
    password: str


@router.post("/register")
async def register(request: UserRequest, db: Session = Depends(get_db)):
    hashed_password = bcrypt.hashpw(
        request.password.encode('utf-8'), bcrypt.gensalt())
    embed = json.dumps([0] * 512)
    # 如果表中已经存在该用户，则返回错误
    if db.query(User).filter(User.number == request.number).first():
        raise HTTPException(status_code=400, detail="User already exists")
    user = User(number=request.number, usertype=request.usertype, name=request.name,
                password=hashed_password.decode('utf-8'), embed=embed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return JSONResponse(status_code=201, content={"message": "User registered successfully"})


@router.post("/login")
async def login(request: UserRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.number == request.number).first()
    if user and bcrypt.checkpw(request.password.encode('utf-8'), user.password.encode('utf-8')):
        token = jwt.encode({"number": request.number, "usertype": user.usertype, "name": user.name,
                           "exp": datetime.now() + timedelta(days=7)}, settings.SECRET_KEY, algorithm="HS256")
        return JSONResponse(status_code=200, content={"token": token, "usertype": user.usertype, "name": user.name})
    raise HTTPException(status_code=401, detail="Authentication failed")


@router.post("/image")
async def upload_image(file: UploadFile = File(...), user: dict = Depends(authenticate_token)):
    if file:
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        file.filename = f"{user['number']}-{current_date}.jpg"
        image_path = os.path.join(
            settings.UPLOAD_DIR, 'image', 'before', file.filename)
        with open(image_path, "wb") as buffer:
            buffer.write(file.file.read())
        result = run_script('face/script.py', image_path)
        result['imageUrl'] = f"http://{get_ip()}:{settings.PORT}/{result['imageUrl']}"
        return JSONResponse(status_code=200, content=result)
    raise HTTPException(status_code=400, detail="No image received")


@router.post("/video")
async def upload_video(file: UploadFile = File(...), user: dict = Depends(authenticate_token)):
    if file:
        file.filename = f"{user['number']}-{user['name']}.mp4"
        video_path = os.path.join(settings.UPLOAD_DIR, 'video', file.filename)
        with open(video_path, "wb") as buffer:
            buffer.write(file.file.read())
        thumbnail_path = os.path.join(
            settings.UPLOAD_DIR, 'metaface', f"{user['number']}.png")
        try:
            ffmpeg.input(video_path).output(thumbnail_path, vframes=1).run()
            return Response(status_code=200)
        except ffmpeg.Error as e:
            raise HTTPException(
                status_code=500, detail=f"Error taking screenshot: {e}")
    raise HTTPException(status_code=400, detail="No video received")


@router.get("/validateToken")
async def validate_token(user: dict = Depends(authenticate_token)):
    return JSONResponse(status_code=200, content={"valid": True, "user": user})


@router.get("/handshake")
async def handshake():
    return Response(status_code=200, content="handshake_ack")


# 存储上次的时间戳
last_timestamp = ""

# 读取最新的统计数据


def read_latest_stats():
    global last_timestamp
    try:
        with open('attention_log.csv', mode='r') as file:
            csv_reader = csv.reader(file)
            rows = list(csv_reader)
            if rows:
                # 读取最后一行
                last_row = rows[-1]
                timestamp, total_count, concentrated_count, absent_minded_count = last_row
                if timestamp != last_timestamp:
                    last_timestamp = timestamp
                    return {
                        "timestamp": timestamp,
                        "total_count": total_count,
                        "concentrated_count": concentrated_count,
                        "absent_minded_count": absent_minded_count
                    }
    except FileNotFoundError:
        return None
    return None


def kill_process_and_children(process_pid):
    process = psutil.Process(process_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    process = await asyncio.create_subprocess_exec(
        "D:/miniconda3/envs/fer/python.exe", "face/headpose.py",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    await websocket.accept()
    try:
        while True:
            stats = read_latest_stats()
            if stats:
                try:
                    await websocket.send_json(stats)
                except WebSocketDisconnect:
                    break
            await asyncio.sleep(4)  # 使用异步sleep而不是time.sleep
    except WebSocketDisconnect:
        pass
    finally:
        if process.returncode is None:  # 检查子进程是否已经终止
            kill_process_and_children(process.pid)
            await process.wait()  # 等待子进程完全终止
        await websocket.close()  # 关闭WebSocket连接
