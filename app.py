from pickle import load
import streamlit as st

@st.cache_resource
def load_model():
    with open("models/nlp-url-spam.pkl", "rb") as f:
        return load(f)

model = load_model()

class_dict = {
    "0": "No Spam",
    "1": "Spam"
}

st.title("Modelo Spam Predicción")
st.markdown("Power by: **Guillermo Lugo**")
st.divider()

val1 = st.text_input(
    "Ingrese el URL",
    placeholder="https://ejemplo.com"
)

if st.button("Predicción"):
    prediction = str(model.predict([val1])[0])
    pred_class = class_dict[prediction]

    st.divider()
    st.write("🔍 **Resultado:**", pred_class)
    st.divider()
