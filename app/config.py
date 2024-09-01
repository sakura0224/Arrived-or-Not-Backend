# app/config.py

import os


# 用于存放配置信息，如端口号、上传文件目录、数据库URL、密钥等
class Settings:
    PORT: int = int(os.getenv("PORT", 8000))
    current_dir = os.path.dirname(__file__)
    UPLOAD_DIR: str = os.path.join(current_dir, '..', 'library')
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "mysql+mysqlconnector://root:root@localhost:3306/face") # 请根据实际情况修改数据库连接信息
    SECRET_KEY: str = os.getenv("SECRET_KEY", "shanghaiuniversity") # 请根据实际情况修改密钥


settings = Settings()
