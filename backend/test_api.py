from fastapi.testclient import TestClient
import base64
from io import BytesIO
from PIL import Image

from api import app


client = TestClient(app)

def test_post_api():

    img_path = './test_images/2.jpg'

    image = Image.open(img_path)

    buffered = BytesIO()

    image.save(buffered, format='JPEG')

    image_str = base64.b64encode(buffered.getvalue())

    k = 1

    res = client.get('/image_search',
                       json={'image_str': image_str, 'num_results': k})

    print(res)

    assert res.json() == {}
