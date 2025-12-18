from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Phase 1 ML Serving API")

# Include API router
app.include_router(router, prefix="/api")
#app = FastAPI(debug=True)

@app.get("/")
def root():
    return {"status": "ok", "message": "ML Serving API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# uvicorn app.main:app --reload

