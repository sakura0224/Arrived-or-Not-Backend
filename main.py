# main.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes import router
from app.database import Base, engine
from app.config import settings
from app.utils import get_ip

# 创建数据库表
Base.metadata.create_all(bind=engine)

app = FastAPI()
app.include_router(router)

# 挂载静态文件
app.mount("/library", StaticFiles(directory="library"), name="library")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=get_ip(), port=settings.PORT)
