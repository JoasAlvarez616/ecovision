# imshow.py — Plan B: clasifica una carpeta local sin Streamlit
import sys, os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from rules import recomendar

model = tf.keras.models.load_model('model/ecovision.keras')
idx2label = {0: "botella_plastica", 1: "lata",
             2: "no_identificado",  3: "papel_carton"}

folder = sys.argv[1] if len(sys.argv) > 1 else "dataset/test"

for cls_dir in sorted(os.listdir(folder)):
    path_cls = os.path.join(folder, cls_dir)
    if not os.path.isdir(path_cls):
        continue
    print(f"\n=== {cls_dir} ===")
    for fname in sorted(os.listdir(path_cls))[:5]:
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        p = os.path.join(path_cls, fname)
        img = image.load_img(p, target_size=(224, 224))
        arr = preprocess_input(np.expand_dims(image.img_to_array(img), 0))
        preds = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        clase_final, contenedor = recomendar(idx2label[idx], conf)
        print(f"  {fname[:40]:42s} -> {clase_final:18s} ({conf*100:5.1f}%) -> {contenedor}")