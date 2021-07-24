from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import FaceDetection as fd

import base64


app = FastAPI()


class Req(BaseModel):
    uploadedImage: str
    extension: str



@app.post("/facedetection",status_code=200)
def detectface(req:Req):
    ext = req.extension
    with open(f"temp.{ext}","wb") as file:
        file.write(base64.b64decode(req.uploadedImage))
        
        
    output = fd.CheckImage("temp.jpg")
    
    return {"output": output}

if __name__ == "__main__":
    uvicorn.run("main:app",reload=True)
