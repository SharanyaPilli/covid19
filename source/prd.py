import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
class_names = ["Covid19", "Normal", "Pneumonia"]
m1=load_model("D:\\DEMO-COVID\\Data\\covid_pneu_model.h5")
img=image.load_img("D:\\DEMO-COVID\\Data\\test\\COVID19\\COVID19(463).jpg",target_size=(224,224))
imgg=image.img_to_array(img)
imgg=np.expand.dims(imgg)
prd=m1.predict(imgg)
ind=np.argmax(prd[0])
print(prd)
