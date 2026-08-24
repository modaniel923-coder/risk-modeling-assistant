"""
后端启动入口
运行: python main.py
访问: http://localhost:8080
"""
import uvicorn
from api.routes import app

if __name__ == "__main__":
    print("=" * 60)
    print("  Risk Modeling Assistant API Server")
    print("  Version: 0.1.0")
    print("  URL: http://localhost:8080")
    print("  API Docs: http://localhost:8080/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8080)
