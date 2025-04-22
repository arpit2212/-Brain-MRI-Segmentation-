import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import cv2

plt.style.use("ggplot")

# Metrics
def dice_coefficients(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

def dice_coefficients_loss(y_true, y_pred, smooth=100):
    return -dice_coefficients(y_true, y_pred, smooth)

def iou(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    summation = K.sum(y_true + y_pred)
    return (intersection + smooth) / (summation - intersection + smooth)

def jaccard_distance(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    return -iou(y_true_flatten, y_pred_flatten)

# App UI
st.title("🧠 Mulltiple Scelerosis WEB-App")

# Load model (local file)
model = load_model("unet_brain_mri_seg.hdf5", custom_objects={
    'dice_coef_loss': dice_coefficients_loss,
    'iou': iou,
    'dice_coef': dice_coefficients
})

im_height = 256
im_width = 256

file = st.file_uploader("📁 Upload MRI Images", type=["png", "jpg"], accept_multiple_files=True)

if file:
    for i in file:
        st.header("🖼️ Original Image:")
        st.image(i)
        content = i.getvalue()
        image = np.asarray(bytearray(content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        img2 = cv2.resize(image, (im_height, im_width))
        img3 = img2 / 255.0
        img4 = img3[np.newaxis, :, :, :]

        if st.button(f"🧠 Predict Segmentation for {i.name}"):
            pred_img = model.predict(img4)[0]
            st.header("🧪 Predicted Segmentation:")
            st.image(pred_img, clamp=True)
