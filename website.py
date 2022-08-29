import sklearn
import numpy as np
import pickle
import streamlit as st
from PIL import Image
# print('The scikit-learn version is {}.'.format(sklearn.__version__))
m = pickle.load(open('lm.pkl','rb'))
lp = pickle.load(open('lp.pkl','rb'))

st.title("Laptop Price Predictor")

img=Image.open('laptop_image.jpeg')
st.image(img)
aud = open("laptop_audio.mpeg", "rb")
st.audio(aud)
comp = st.selectbox('Company',lp['Company'].unique())
type = st.selectbox('Type',lp['TypeName'].unique())
os = st.selectbox('OS',lp['OpSys'].unique())
cpu = st.selectbox('CPU',lp['CPU_brand'].unique())
RAM = st.selectbox('RAM(GB)',[2,4,6,8,12,16,24,32,64])
wt = st.number_input('Weight of the Laptop')
ips = st.selectbox('IPS Display',['No','Yes'])
ts = st.selectbox('Touch Screen Display',['No','Yes'])
ssd = st.selectbox('SSD (GB)',[0,8,128,256,512,1024])
hdd = st.selectbox('HDD (GB)',[0,128,256,512,1024,2048])
flash = st.selectbox('Flash Memory',[0,128,256,32,64,16,512])
hybrid=st.selectbox('Hybrid',[0,1000,508])
gpu = st.selectbox('GPU',lp['GPU_Company'].unique())
ss = st.number_input('Screen Size')
res = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
clk_speed = st.number_input('Clock Speed (GHz)')

if st.button('Predict Price'):
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    if ts == 'Yes':
        ts = 1
    else:
        ts = 0

    X_res = int(res.split('x')[0])
    Y_res = int(res.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/ss
    query = np.array([comp,type,cpu,gpu,os,wt,ss,RAM,hdd,ssd,hybrid,flash,ips,ts,ppi,clk_speed])

    query = query.reshape(1,16)
    st.title("Estimated Price : " + str(int(np.exp(m.predict(query)[0]))) + " Euros")
