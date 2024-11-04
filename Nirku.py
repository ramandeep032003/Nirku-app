import streamlit as st
import numpy as np
import tensorflow as tf
import psycopg2
from PIL import Image
from pathlib import Path
model_path=Path('C:/Users/deepr/Desktop/vscode/best_model6.h5')

try:
    best_model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    best_model = None 
def connect_to_db():
    return psycopg2.connect(
        dbname="patient_db",       # Your database name
        user="postgres",      # Your PostgreSQL username
        password="raman123",  # Your PostgreSQL password
        host="localhost",          # Database server address
        port="5432"                # Default PostgreSQL port
    )

def preprocess_image(image):
    """Resize and normalize the image for model input."""
    img = Image.fromarray(image)  # Convert to Pillow Image
    img = img.resize((224, 224))  # Resize the image
    img = np.array(img)  # Convert BGR to RGB
    img = img / 255.0  # Normalize the image
    return img.reshape((1, 224, 224, 3))  # Add batch dimension

def predict_pneumonia(image):
    """Predict pneumonia from the uploaded X-ray image."""
    input_data = preprocess_image(image)
    prediction = best_model.predict(input_data)
    binary_predictions = (prediction > 0.5).astype(int)

    # Interpret the prediction (0: No Pneumonia, 1: Pneumonia)
    result = "Pneumonia" if binary_predictions[0][0] == 1 else "No Pneumonia"
    return result

# Streamlit UI
st.markdown(
    """
    <style>
    .stApp {
       background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
    }
    .main {
        background-color: #d0f0c0;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Nirku: Your X-Ray Companion for Pneumonia Diagnosis")
st.header("Pneumonia Detection from X-Ray Images")
id = st.number_input("ID", min_value=1) 
patient_id = st.number_input("Patient ID", min_value=1) 
patient_name = st.text_input("Patient Name")
patient_age = st.number_input("Patient Age", min_value=0, max_value=120)
patient_gender = st.selectbox("Patient Gender", options=["Male", "Female", "Other"])
visit_date = st.date_input("Visit Date")
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])


if st.button("Predict"):
    if uploaded_file is not None:
        # Read and process the image
        image = Image.open(uploaded_file).convert("RGB")
        image = np.array(image)  # Convert to NumPy array
      

        result = predict_pneumonia(image)
        
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute('''
                INSERT INTO patients_data (id,patient_id,name, age, gender, date,prediction)
                VALUES (%s,%s,%s, %s, %s,%s, %s)
            ''', (id,patient_id,patient_name, patient_age, patient_gender,visit_date, result))
        conn.commit()
        cursor.close()
        conn.close()

        st.image(image, use_column_width=True)
        st.write(f"Patient id: {patient_id}")
        st.write(f"Patient Name: {patient_name}")
        st.write(f"Patient Age: {patient_age}")
        st.write(f"Patient Gender: {patient_gender}")
        st.write(f"visit_date: {visit_date}")
        st.write(f"Prediction: {result}")
        if result == "Pneumonia":
            st.warning("### Possible Solutions for Pneumonia:")
            st.write("1. **Consult a Healthcare Professional**: It's crucial to schedule an appointment for a comprehensive evaluation.")
            st.write("2. **Possible Treatment Options**: Depending on the cause, you may require medications. Consult your doctor.")
            st.write("3. **Home Care Recommendations**: Ensure you rest adequately and stay hydrated.")
            st.write("4. **Warning Signs to Watch For**: Monitor for difficulty breathing, persistent fever, or chest pain.")
            st.write("5. **Follow-Up Actions**: You may need additional tests as advised by your doctor.")
            st.write("6. **Preventive Measures**: Discuss vaccinations with your doctor to prevent future infections.")
            st.write("7. **Lifestyle Changes**: Adopt a healthy lifestyle to strengthen your immune system.")
            st.write(" THANKS FOR USING NIRKU PNEUMONIA PREDICTION APP")
    else:
        st.error("Please upload an image and enter the image name.")
    
