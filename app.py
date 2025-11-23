import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import urllib.request
import os
import ssl

st.set_page_config(page_title="Detector Eterno", layout="centered")
st.title("👁️ Vigilância Permanente")

# --- Download Anti-Falha ---
def download_files():
    ssl._create_default_https_context = ssl._create_unverified_context
    files = ["MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel"]
    mirrors = {
        "MobileNetSSD_deploy.prototxt": ["https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt"],
        "MobileNetSSD_deploy.caffemodel": ["https://github.com/djmv/MobilNet_SSD_opencv/raw/master/MobileNetSSD_deploy.caffemodel"]
    }
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for f in files:
        if not os.path.exists(f):
            for url in mirrors[f]:
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req) as resp, open(f, 'wb') as out:
                        out.write(resp.read())
                    if os.path.getsize(f) > 100: break
                except: continue

download_files()

CLASSES = ["fundo", "aviao", "bicicleta", "passaro", "barco", "garrafa", "onibus", "carro", "gato", "cadeira", "vaca", "mesa", "cachorro", "cavalo", "moto", "PESSOA", "planta", "ovelha", "sofa", "trem", "tv"]
COLOR_RED = (0, 0, 255)
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(img, (startX, startY), (endX, endY), COLOR_RED, 2)
                label = f"{CLASSES[idx]}: {confidence*100:.0f}%"
                cv2.putText(img, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 2)
        return img

webrtc_streamer(key="camera", mode=WebRtcMode.SENDRECV, 
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
                video_processor_factory=VideoProcessor, async_processing=True)
