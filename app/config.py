# app/config.py

import os


class Settings:
    PORT: int = int(os.getenv("PORT", 8000))
    current_dir = os.path.dirname(__file__)
    UPLOAD_DIR: str = os.path.join(current_dir, '..', 'library')
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "mysql+mysqlconnector://root:root@localhost:3306/face")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "shanghaiuniversity")


settings = Settings()
