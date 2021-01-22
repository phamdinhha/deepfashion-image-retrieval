import streamlit as st
import time
import streamlit.components.v1 as components
from PIL import Image
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import io

st.title('AI-Stylist System')
menu = ['Image Retrieval', 'Clothes Try-on', 'Fashion compatilibity and recomendation']
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Image Retrieval':
    # hostname = socket.gethostname()
    # ip = socket.gethostbyname(hostname)
    backend = 'http://0.0.0.0:8080/imagesearch'

    html_temp = """
                    <div style="background-color:royalblue;padding:10px;border-radius:10px">
                    <h1 style="color:white;text-align:center;font-family:Verdana">Fashion Image Retrieval Engine</h1>
                    </div>
                """
    components.html(html_temp)

    query_image = st.file_uploader("Upload an image to retrive relevant fashion product", type="jpg")

    if query_image:
        image = Image.open(query_image)
        st.image(image, caption='Query Image', width=300)
        k = st.number_input('Please set the number of images to retrieve', 1, 15, step=1, key='k')
        if st.button('Get similar clothes'):
            m = MultipartEncoder(fields={'file': ('filename', query_image, 'image/jpeg'),
                                        'k': str(k)})
            res = requests.post(backend, 
                                data=m, 
                                headers={"Content-Type": m.content_type}, 
                                timeout=8000)
            response = res.json()
            col1, col2 = st.beta_columns(2)
            for i in range(len(response)):
                if i%2 == 0:
                    col1.image(Image.open(response[i][0]), 
                                        width=250, 
                                        caption='Confidence score: '+str(response[i][1]), 
                                        use_column_width=True)
                else:
                    col2.image(Image.open(response[i][0]), 
                                        width=250, 
                                        caption='Confidence score: '+str(response[i][1]), 
                                        use_column_width=True)
            with st.spinner('Wait for it ...'):
                time.sleep(0.1)
            st.success('Done!!!')

    else: 
        st.text('Please upload an image to retrieve similar fashion product')

if choice == 'Clothes Try-on':
    backend = 'http://0.0.0.0:8081/tryon'
    html_temp = """
                    <div style="background-color:royalblue;padding:10px;border-radius:10px">
                    <h1 style="color:white;text-align:center;font-family:Verdana">Fashion Product Try-on Engine</h1>
                    </div>
                """
    components.html(html_temp)

    image = st.file_uploader("Image", type="jpg")
    image_parse = st.file_uploader("Image parse", type="png")
    cloth = st.file_uploader("Cloth", type="jpg")
    cloth_mask = st.file_uploader("Cloth_mask", type="jpg")
    pose_file = st.file_uploader("Pose", type="json")

    if image and image_parse and cloth and cloth_mask and pose_file:
        col1, col2, col3, col4 = st.beta_columns(4)
        col1.image(Image.open(image), width=200, caption='Image', use_column_width=True)
        col2.image(Image.open(image_parse).convert('RGB'), width=200, caption='Image parse', use_column_width=True)
        col3.image(Image.open(cloth), width=200, caption='Cloth', use_column_width=True)
        col4.image(Image.open(cloth_mask), width=200, caption='Cloth mask', use_column_width=True)
        if st.button('Try this cloth'):
            m = MultipartEncoder(fields={'image': ('filename', image, 'image/jpg'),
                                         'image_parse': ('filename', image_parse, 'image/png'),
                                         'cloth': ('filename', cloth, 'image/jpg'),
                                         'cloth_mask': ('filename', cloth_mask, 'image/jpg'),
                                         'pose': ('filename', pose_file, 'json')})
            res = requests.post(backend,
                                data=m, 
                                headers={"Content-Type": m.content_type}, 
                                timeout=8000)

            output_image = Image.open(io.BytesIO(res.content)).convert("RGB")
            st.image(output_image, width=500, caption='Try-on output.')
            with st.spinner('Wait for it ...'):
                time.sleep(0.1)

    else: 
        st.text('Please upload an image to retrieve similar fashion product')

if choice == 'Fashion compatilibity and recomendation':
    backend = ''
    html_temp = """
                    <div style="background-color:royalblue;padding:10px;border-radius:10px">
                    <h1 style="color:white;text-align:center;font-family:Verdana">Fashion Recommendation Engine</h1>
                    </div>
                """
    components.html(html_temp)

    query_image = st.file_uploader("Upload an image to get the recommendations", type="jpg")

    if query_image:
        image = Image.open(query_image)
        st.image(image, caption='Query Image', width=300)
        if st.button('Get recommendations'):
            m = MultipartEncoder(fields={'file': ('filename', query_image, 'image/jpeg')})
            res = requests.post(backend,
                                data=m, 
                                headers={"Content-Type": m.content_type}, 
                                timeout=8000)
            response = res.json()
            col1, col2 = st.beta_columns(2)
            for i in range(len(response)):
                col1.image(Image.open(response[i][0]), 
                                      width=250, 
                                      caption='Confidence score: '+str(response[i][1]), 
                                      use_column_width=True)
                col2.image(Image.open(response[i][0]), 
                                      width=250, 
                                      caption='Confidence score: '+str(response[i][1]), 
                                      use_column_width=True)
            with st.spinner('Wait for it ...'):
                time.sleep(0.1)
            st.success('Done!!!')

    else: 
        st.text('Please upload an image to retrieve similar fashion product')