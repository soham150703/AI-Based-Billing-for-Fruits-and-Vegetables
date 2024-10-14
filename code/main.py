import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Load the model
model = load_model('fine_tuned_model2.h5')

# Load class labels
with open('label.txt') as f:
    class_labels = [line.strip() for line in f]

# Define the prices per 100g for all 36 items
prices = {
    "apple": 50,
    "banana": 30,
    "beetroot": 40,
    "bell pepper": 60,
    "cabbage": 20,
    "capsicum": 50,
    "carrot": 30,
    "cauliflower": 35,
    "chilli pepper": 70,
    "corn": 25,
    "cucumber": 15,
    "eggplant": 30,
    "garlic": 200,
    "ginger": 150,
    "grapes": 90,
    "jalepeno": 60,
    "kiwi": 120,
    "lemon": 10,
    "lettuce": 25,
    "mango": 80,
    "onion": 20,
    "orange": 40,
    "paprika": 100,
    "pear": 50,
    "peas": 60,
    "pineapple": 45,
    "pomegranate": 150,
    "potato": 20,
    "radish": 25,
    "soy beans": 70,
    "spinach": 30,
    "sweetcorn": 30,
    "sweetpotato": 40,
    "tomato": 25,
    "turnip": 30,
    "watermelon": 20,
}

def predict_class(img_array):
    # Preprocess the image
    img_array = preprocess_input(img_array)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    return class_labels[predicted_class_index]

# Streamlit sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction", "Terminate Project"])

# Main Page
if app_mode == "Home":
    st.header("Fruits and Vegetable Classification System")
    st.image("D:/AI_CP/image.png")

# About Project
elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset has images regarding fruits and vegetables.")
    st.text("Fruits - Apple, Banana, etc.")
    st.text("Vegetables - Beetroot, Cabbage, etc.")
    st.subheader("Content")
    st.text("Train, Test, and Validation datasets are used.")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")

    # Initialize session state for the bill
    if 'bill' not in st.session_state:
        st.session_state['bill'] = []

    # Camera Input
    st.subheader("Capture Image from Camera")
    camera_image = st.camera_input("Take a picture")

    if camera_image:
        # Load the image from the file-like object
        image_pil = Image.open(io.BytesIO(camera_image.read()))
        image_pil = image_pil.resize((224, 224))  # Resize to the required size for MobileNetV2
        
        # Display the image
        st.image(image_pil, caption='Uploaded Image', use_column_width=True)
        
        # Convert the image to an array
        img_array = image.img_to_array(image_pil)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predicted_class = predict_class(img_array)
        st.success(f"The model predicts this image is a: {predicted_class}")

        # Get the weight input from the user
        weight = st.number_input(f"Enter the weight of {predicted_class} in grams", min_value=0.0, step=0.1)

        if weight > 0:
            price_per_100g = prices.get(predicted_class.lower(), 0)
            total_cost = (price_per_100g / 100) * weight
            st.write(f"Price per 100g: ₹{price_per_100g}")
            st.write(f"Total cost for {weight}g: ₹{total_cost:.2f}")

            # Store the result for billing
            st.session_state.bill.append((predicted_class, weight, total_cost))

    # Display the total bill if items exist
    if st.session_state['bill']:
        st.subheader("Total Bill")
        total_amount = 0
        for item, wt, cost in st.session_state.bill:
            st.write(f"{item}: {wt}g - ₹{cost:.2f}")
            total_amount += cost
        st.write(f"**Total Amount: ₹{total_amount:.2f}**")

# Terminate Project
elif app_mode == "Terminate Project":
    st.header("Terminating the Project")
    if st.button("Confirm Termination"):
        # Clear the session state
        st.session_state.clear()
        st.write("The project has been terminated. All session data has been cleared.")
