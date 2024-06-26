# app/utils.py

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings
import subprocess
import json
import os
from rmn import RMN

security = HTTPBearer()


def fer():
    m = RMN()
    m.video_demo()


def authenticate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=403, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=403, detail="Invalid token")


def run_script(script, image_path):
    process = subprocess.Popen(['D:/miniconda3/envs/fer/python.exe', script, image_path],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise Exception(f"Error processing image: {stderr.decode('utf-8')}")

    try:
        output = json.loads(stdout.decode('utf-8'))
        return output
    except json.JSONDecodeError:
        raise Exception("Error parsing script output")

def get_ip():
    import psutil
    import socket

    ethernet_ip = None
    wifi_ip = None

    for interface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and not addr.address.startswith('127.') and not addr.address.startswith('169.254'):
                if '以太网' in interface and ethernet_ip is None:
                    ethernet_ip = addr.address
                elif 'WLAN' in interface and wifi_ip is None:
                    wifi_ip = addr.address

    return wifi_ip or ethernet_ip
