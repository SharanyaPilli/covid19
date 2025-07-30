import streamlit as st
import numpy as np
import joblib
import requests
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing import image
import google.generativeai as genai

genai.configure(api_key="AIzaSyB3gA1UpLRB-fp16Jz4GY7iOg9_4v67Rvw")
model = genai.GenerativeModel(model_name="gemini-2.0-flash")
chat = model.start_chat(history=[])

ml = load_model("D:\\DEMO-COVID\\Data\\covid_pneu_model.h5")

classes = ["Covid19", "Normal", "Pneumonia"]  # define classes here

st.title("pneumonia/Covid Diagnosis System 🫁")
upl = st.file_uploader("upload the image", type=["png","jpg","jpeg"])

if upl:
    img = Image.open(upl).convert("RGB")
    img = img.resize((224,224))
    st.image(img, use_container_width=True)
    img = image.img_to_array(img)
    imgg = np.expand_dims(img, axis=0)/255.0
    prd = ml.predict(imgg)
    res = np.argmax(prd)
    class_name = classes[res]   # use 'classes', not undefined 'classes'
    st.write(class_name)
    
    if class_name == "Normal":
        st.success("you are healthy")
    else:
        st.subheader("Diagnosis 🩺")
        Age = st.slider("Age",0,80)
        Gn = st.selectbox("Gender",["Male","Female"])
        fv = st.selectbox("Fever (Yes/No)",["Yes","No"])
        cf = st.selectbox("Cough (Yes/No)",["Yes","No"])
        ft = st.selectbox("Fatigue (Yes/No)",["Yes","No"])
        brt = st.selectbox("Breathlessness (Yes/No)",["Yes","No"])
        cm = st.selectbox("Commorbidity (Yes/No)",["Yes","No"])
        stt = st.selectbox("Stage",["mild","moderate","severe"])
        tp = st.selectbox("Type",["viral","bacteria"])
        ts = st.slider("Tumor_size",0,5)
        
        gn_num = 1 if Gn == "Male" else 0
        fv_num = 1 if fv == "Yes" else 0
        cf_num = 1 if cf == "Yes" else 0
        ft_num = 1 if ft == "Yes" else 0
        brt_num = 1 if brt == "Yes" else 0
        cm_num = 1 if cm == "Yes" else 0
        stt_num = 0 if stt == "mild" else 1 if stt == "moderate" else 2
        tp_num = 0 if tp == "viral" else 1
        
        data_inp = {
            "Age": Age,
            "Gender": gn_num,
            "Fever": fv_num,
            "Cough": cf_num,
            "Fatigue": ft_num,
            "Breathlessness": brt_num,
            "Commorbidity": cm_num,
            "Stage": stt_num,
            "Type": tp_num,
            "Tumor_Size": ts
        }
        
        res = requests.post("http://127.0.0.1:8000//predict", json=data_inp)
        response = res.json()
        st.write(response.get("prediction", "No prediction found in response"))

        if "Prediction" in response and float(response["Prediction"]) > 0.5:
            print("Positive")
        else:
            print("Negative or Prediction key not found")
            prompt="Suggest a life style habit to overcome pneumonia or covid" 
            res=chat.send_message(prompt) 
            st.markdown("Suggesions") 
            st.markdown(res.text)
