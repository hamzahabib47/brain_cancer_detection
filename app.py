import numpy as np
import matplotlib.pyplot as plt
import cv2
import streamlit as st
import keras

model = keras.models.load_model('Brain_cancer_Detection.h5')

st.title(":violet[Brain Cancer Detector]")

input_image_path = st.file_uploader('Upload the MRI image file...', type=['jpg', 'jpeg'])

if st.button('Detect'):
    if input_image_path == None:
        st.header('Please upload the file...')
    else:
        try:
            input_image = plt.imread(input_image_path) # or plt.imread()

            plt.imshow(input_image) # or plt.imshow()

            input_image_resize = cv2.resize(input_image, (128, 128))

            input_image_scaled = input_image_resize/255

            image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3])

            input_prediction = model.predict(image_reshaped)

            input_pred_label = np.argmax(input_prediction)

            if input_pred_label == 1:
                st.header(":red[Yes! Brain Cancer Detected]")
                st.image(input_image_path)
            else:
                st.header(":green[No Cancer Detected!]")
                st.image(input_image_path)
        except:
            st.header("An Error Occured! Please try again or select another file.")