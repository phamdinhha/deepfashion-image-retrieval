from io import BytesIO
from pandas.core import base
from retriever import retrieve
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile, Form
from starlette.responses import JSONResponse
from PIL import Image
import base64

app = FastAPI()

@app.post('/imagesearch')
def retrieve_image(file: UploadFile = File(...), k: int = Form(...)):
    image = Image.open(file.file)
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    image_str = base64.b64encode(buffered.getvalue())
    results = retrieve(image_str, True, k)
    return JSONResponse(status_code=200, content=results)

if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0', port=8080)