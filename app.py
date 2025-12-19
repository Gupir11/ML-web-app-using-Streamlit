from pickle import load
import streamlit as st


model = load(open("models/nlp-url-spam.pkl", "rb"))
vectorizer = load(open("data/nlp-url-spam-vectorizer", "rb"))
class_dict = {"0": "No Spam",
              "1": "Spam"}

st.title("Modelo Spam Prediccion")
st.markdown("""Power by: [Guillermo Lugo]""")
st.divider()

val1 = st.text_input(label, value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder="ingrese el URL", disabled=False, label_visibility="visible", icon=None, width="stretch")


if st.button("Prediccion"):
    prediction = str(model.predict([[val1]])[0])
    pred_class = class_dict[prediction]
    st.divider()
    st.write("Prediction:", pred_class)
    st.divider()