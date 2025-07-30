import google.generativeai as genai
genai.configure(api_key="AIzaSyB3gA1UpLRB-fp16Jz4GY7iOg9_4v67Rvw")
model=genai.GenerativeModel(model_name="gemini-2.0-flash")
chat=model.start_chat(history=[])
while True:
    prmt=input()
    if(prmt=="exit"):
        break
    res=chat.send_message(prmt)
    print(res.text)