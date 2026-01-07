from pickle import load
import streamlit as st

model = load(open("models/nlp-url-spam.pkl", "rb"))
vectorizer = load(open("models/nlp-url-spam-vectorizer.pkl", "rb"))
class_dict = {0: "No Spam", 1: "Spam"}

st.title("Spam - Model prediction")
st.markdown("""Power by: [Guillermo Lugo]""")
st.divider()

url = st.text_input("Enter a URL to analyze")

if st.button("Predict"):
    url_vec = vectorizer.transform([url]).toarray()
    prediction = model.predict(url_vec)[0]
    pred_class = class_dict[prediction]
    st.divider()
    st.write("Prediction:", pred_class)
    st.divider()
