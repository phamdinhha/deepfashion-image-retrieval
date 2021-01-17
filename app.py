import streamlit as st
import time
import streamlit.components.v1 as components
from PIL import Image
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

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
        res = requests.post(backend, data=m, headers={"Content-Type": m.content_type}, timeout=8000)
        response = res.json()
        col1, col2 = st.beta_columns(2)
        for i in range(len(response)):
            if i%2 == 0:
                col1.image(Image.open(response[i][0]), width=250, caption='Confidence score: '+str(response[i][1]), use_column_width=True)
            else:
                col2.image(Image.open(response[i][0]), width=250, caption='Confidence score: '+str(response[i][1]), use_column_width=True)
        with st.spinner('Wait for it ...'):
            time.sleep(0.1)
        st.success('Done!!!')

else: 
    st.text('Please upload an image to retrieve similar fashion product')
