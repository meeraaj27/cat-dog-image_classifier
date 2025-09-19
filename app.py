import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import zipfile
import tempfile
import matplotlib.pyplot as plt

#  Page setup 
st.set_page_config(page_title="Image Classifier project", layout="wide")
st.title("Meera's Cats-Dogs-Classifier")
st.write("Upload an image or take a photo to classify it as a cat or dog!")

# Load trained model 
MODEL_PATH = "final_image_classifier.keras"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file {MODEL_PATH} not found!")
else:
    model = tf.keras.models.load_model(MODEL_PATH)



IMG_SIZE = (160, 160)
class_names = ["Cat", "Dog"]

# Track model status 
if "model_status" not in st.session_state:
    st.session_state.model_status = "Original Model"
    st.session_state.last_retrained_file = None

if st.session_state.model_status == "Original Model":
    st.success("Currently using: Original Model")
else:
    st.warning("Currently using: Retrained Model")



# Camera button 
st.markdown("### Image Input Options")
if "use_camera" not in st.session_state:
    st.session_state.use_camera = False

def toggle_camera():
    st.session_state.use_camera = not st.session_state.use_camera

st.button("ON/OFF CAMERA", on_click=toggle_camera)
use_camera = st.session_state.use_camera

img = None



# Input section 
if use_camera:
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        img = Image.open(camera_image).convert("RGB")
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")



# Display images  & Prediction 
if img is not None:
    display_size = (300, 300)  

    # Original image 
    display_img = img.resize(display_size)
    st.image(display_img, caption="Captured Image")

    # Preprocessed image 
    resized_img = img.resize(IMG_SIZE)         
    show_img = resized_img.resize(display_size)  
    st.image(show_img, caption="Preprocessed Image ")


    # Prediction
    img_array = np.array(resized_img)
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]
    confidence = pred if pred > 0.5 else 1 - pred
    label = class_names[1] if pred > 0.5 else class_names[0]

   
    if st.session_state.model_status == "Original Model":
        st.subheader("Prediction Results ")
    else:
        st.subheader("Prediction Results (Retrained Model)")

    st.write(f"Prediction: {label}")
    st.write(f"Confidence: {confidence*100:.2f}%")



#  Performance statistics 
st.markdown("---")
st.subheader("Training Statistics")
if os.path.exists("training_curves.png"):
    st.image("training_curves.png", caption="Training vs Validation Curves", width=800)


#  Retraining 
st.markdown("---")
st.subheader("Retrain Model with New Examples")

# Reset button
if st.button("Reset to Original Model"):
    model = tf.keras.models.load_model(MODEL_PATH)
    st.session_state.model_status = "Original Model"
    st.session_state.last_retrained_file = None
    st.success("Switched back to original model!")

retrain_folder = st.file_uploader(
    "Upload a ZIP folder with new training images (cats/dogs subfolders)", type="zip"
)

if retrain_folder is not None and retrain_folder.name != st.session_state.last_retrained_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(retrain_folder, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        st.success("New dataset uploaded and extracted!")

        # Create train/validation split (80% train, 20% val)
        train_ds = tf.keras.utils.image_dataset_from_directory(
            tmpdir, image_size=IMG_SIZE, batch_size=4, label_mode="int",
            validation_split=0.2, subset="training", seed=42
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            tmpdir, image_size=IMG_SIZE, batch_size=4, label_mode="int",
            validation_split=0.2, subset="validation", seed=42
        )

        with st.spinner("Retraining model on new examples..."):
            #  data augmentation layers
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
            ])

            # Clone the original model
            temp_model = tf.keras.models.clone_model(model)
            temp_model.set_weights(model.get_weights())

            # Create a new model that chains augmentation and the original model
            model_to_train = tf.keras.Sequential([
                data_augmentation,
                temp_model
            ])

         
            model_to_train.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            history = model_to_train.fit(train_ds, validation_data=val_ds, epochs=8)


        # Switch to retrained model
        model = temp_model
        st.session_state.model_status = "Retrained Model"
        st.session_state.last_retrained_file = retrain_folder.name
        st.success("Retraining complete. Now using retrained model for predictions!")

        #  Fresh Performance Statistics 
        train_acc = history.history["accuracy"][-1]
        val_acc = history.history["val_accuracy"][-1]
        train_loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]

        st.subheader("Retrained Model Performance Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Accuracy", f"{train_acc*100:.2f}%")
            st.metric("Training Loss", f"{train_loss:.4f}")
        with col2:
            st.metric("Validation Accuracy", f"{val_acc*100:.2f}%")
            st.metric("Validation Loss", f"{val_loss:.4f}")
        
        # performance
        if val_acc > 0.9:
            st.success(" Great performance! The model is very accurate.")
        elif val_acc > 0.7:
            st.warning("Moderate performance. Can improve with more data.")
        else:
            st.error("Low performance. Retrain with more examples.")

        # Save retrained model 
        save_retrain = st.checkbox("Save retrained model ")
        if save_retrain:
            model.save("final_image_classifier_retrained.keras")
            st.success("Retrained model saved as final_image_classifier_retrained.keras!")