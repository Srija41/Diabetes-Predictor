import gradio as gr
import numpy as np
import tensorflow as tf

# Load the saved Keras model
model = tf.keras.models.load_model("diabetes_model.keras")

# Prediction function
def predict_diabetes(pregnancies, glucose, bp, skin, insulin, bmi, dpf, age):
    input_dict = {
        'Pregnancies': np.array([[pregnancies]]),
        'Glucose': np.array([[glucose]]),
        'BloodPressure': np.array([[bp]]),
        'SkinThickness': np.array([[skin]]),
        'Insulin': np.array([[insulin]]),
        'BMI': np.array([[bmi]]),
        'DiabetesPedigreeFunction': np.array([[dpf]]),
        'Age': np.array([[age]])
    }

    prediction = model.predict(input_dict)[0][0]
    label = "Diabetes" if prediction > 0.5 else "No Diabetes"
    confidence = f"{prediction * 100:.2f}%"
    
    return f"{label} (Confidence: {confidence})"

# Gradio interface
iface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Slider(0, 17, step=1, label="Pregnancies"),
        gr.Slider(0, 200, label="Glucose"),
        gr.Slider(0, 122, label="Blood Pressure"),
        gr.Slider(0, 100, label="Skin Thickness"),
        gr.Slider(0, 846, label="Insulin"),
        gr.Slider(0.0, 67.1, label="BMI"),
        gr.Slider(0.0, 2.5, label="Diabetes Pedigree Function"),
        gr.Slider(21, 90, label="Age")
    ],
    outputs="text",
    title="🩺 Diabetes Risk Predictor",
    description="Input patient data to predict whether they are likely to have diabetes."
)

iface.launch()

