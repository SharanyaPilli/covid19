import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
df=pd.read_csv("D:\DEMO-COVID\Data\pneumonia_covid_diagnosis_dataset.csv")
print(df.head())
columns=["Gender","Fever","Cough","Fatigue","Breathlessness","Comorbidity","Type","Stage"]
for col in columns:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
df=df.drop("Is_Curable",axis=1)
x=df.drop(columns=["Survival_Rate"])
y=df["Survival_Rate"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestRegressor()
model.fit(x_train,y_train)
prd=model.predict(x_test)
joblib.dump(model,"covid_diag.pkl")