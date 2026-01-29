"""
run.py - Simple launcher for FaceTales server
"""
import uvicorn

if __name__ == "__main__":
    print("Starting FaceTales server...")
    print("Open http://127.0.0.1:7860 in your browser")
    uvicorn.run("server:app", host="127.0.0.1", port=7860, reload=True)
