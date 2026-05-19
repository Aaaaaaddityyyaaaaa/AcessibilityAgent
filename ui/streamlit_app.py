import streamlit as st
import requests
import base64

st.set_page_config(page_title="VisionAI", page_icon="🤖")
st.title("VisionAI — Accessibility Assistant")

API_URL = "https://aaaadddittyyaaa-agent-space.hf.space/Agent"

mode = st.radio("Mode", ["Caption Image", "Ask Question"], horizontal=True)

image_b64 = None

if mode == "Caption Image":
    uploaded = st.file_uploader("Upload image", type=["jpg", "png", "webp"])
    if uploaded:
        st.image(uploaded, caption="Uploaded Image")
        image_b64 = base64.b64encode(uploaded.read()).decode()

    if st.button("Generate Caption"):
        with st.spinner("Processing..."):
            res = requests.post(API_URL, json={"image": image_b64, "prompt": ""})
            data = res.json()
            if data.get("text"):
                st.success(data["text"])
            if data.get("audio"):
                audio_bytes = base64.b64decode(data["audio"])
                st.audio(audio_bytes, format="audio/mp3")

else:
    prompt = st.text_input("Ask a question")

    if st.button("Ask Agent"):
        with st.spinner("Thinking..."):
            res = requests.post(API_URL, json={"image": None, "prompt": prompt})
            data = res.json()
            if data.get("text"):
                st.success(data["text"])
            if data.get("audio"):
                audio_bytes = base64.b64decode(data["audio"])
                st.audio(audio_bytes, format="audio/mp3")