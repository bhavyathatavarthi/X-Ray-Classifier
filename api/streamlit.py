import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(
    page_title="X-Ray Classifier",
    page_icon="🩺",
    layout="centered"
)


st.title('CHEST X-RAY CLASSIFIER')
st.write("Upload an image of a X-RAY to get its predicted label")
uploaded_files=st.file_uploader(
    "Upload one or more chest x-ray images",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True,
    help="You can drag and drop or select multiple images"
)
if uploaded_files:

    if st.button('Predict'):
        with st.spinner('Analyzing Images...'):
            cols = st.columns(3)
            col_index = 0

            for file in uploaded_files:
                image = Image.open(file).convert("RGB")
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="PNG")
                img_bytes.seek(0)

                files = {"file": ("image.png", img_bytes, "image/png")}
                try:
                    response = requests.post('http://127.0.0.1:8000/predict', files=files)
                    if response.status_code == 200:
                        result=response.json()
                        label=result.get('predicted_label')
                        conf=result.get('confidence_score')

                        with cols[col_index]:
                            st.image(image, caption=f"Uploaded", use_container_width=True)
                            st.markdown(f"**Prediction:** {label}")
                            if conf is not None:
                                st.markdown(f"**Confidence:** {conf * 100:.2f}%")
                            else:
                                st.markdown("Confidence not provided")

                        col_index = (col_index + 1) % 3


                    else:
                        st.error(f"Server returned status code {response.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the FastAPI server. Make sure it is running.")

    else:
        st.info("Upload one or more images to get started")