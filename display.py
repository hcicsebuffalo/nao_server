from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name='images')
@app.get("/", response_class=HTMLResponse)
def serve():
    with open("static/html.txt") as f:
        data = f.read()
    return data