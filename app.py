import streamlit as st
import tensorflow as tf
import numpy as np
import io
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
from rules import recomendar

@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model('model/ecovision.keras')

model = load_model()

idx2label = {0: "botella_plastica", 1: "lata",
             2: "no_identificado",  3: "papel_carton"}

st.set_page_config(page_title="EcoVision", page_icon="E")
st.title("EcoVision - Clasificador de residuos")
st.write("Sube una imagen y te digo que es y en que contenedor va.")
st.markdown("---")

uploaded = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    arr = np.expand_dims(image.img_to_array(img.resize((224, 224))), 0)
    arr = preprocess_input(arr)

    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    clase_nn = idx2label[idx]
    clase_final, contenedor = recomendar(clase_nn, conf)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="Imagen subida", use_column_width=True)
    with col2:
        st.subheader("Resultado")
        st.metric("Red neuronal", clase_nn)
        st.metric("Confianza", f"{conf*100:.1f}%")
        st.metric("Clasificacion final", clase_final)
        st.success(f"Contenedor: {contenedor}")

    if conf < 0.60:
        st.warning("Confianza baja: el objeto podria no ser identificable.")

    with st.expander("Ver probabilidades por clase"):
        for i, p in enumerate(preds):
            st.progress(float(p), text=f"{idx2label[i]}: {p*100:.1f}%")

st.markdown("---")
st.caption("Red neuronal: MobileNetV2. Logica: rules.py")