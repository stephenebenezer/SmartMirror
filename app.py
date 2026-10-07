import streamlit as st
import streamlit.components.v1 as components
import base64
import os
from datetime import datetime

st.set_page_config(page_title="AI Smart Mirror", layout="wide")

st.title("AI-Enabled Smart Mirror")
st.subheader(datetime.now().strftime("%A, %B %d, %Y - %I:%M %p"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_base64_img(filename):
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

outfit1_b64 = get_base64_img("outfit1.png")
outfit2_b64 = get_base64_img("outfit2.png")
outfit3_b64 = get_base64_img("outfit3.png")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Outfit Selection")
    outfit_choice = st.radio(
        "Choose an Outfit:",
        ["None", "Outfit 1", "Outfit 2", "Outfit 3"],
        key="outfit_radio"
    )
    
    st.markdown("---")
    st.header("Mirror Widgets")
    st.write("• **Weather:** 29°C / Partly Cloudy")
    st.write("• **Schedule:** Science Exhibition Demo")

with col2:
    st.header("Live Mirror Feed")
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .container {{
                position: relative;
                width: 100%;
                max-width: 640px;
                margin: auto;
            }}
            #webcam {{
                width: 100%;
                height: auto;
                transform: scaleX(-1);
                border-radius: 10px;
                border: 2px solid #4CAF50;
            }}
            #overlay {{
                position: absolute;
                top: 35%;
                left: 15%;
                width: 70%;
                height: auto;
                pointer-events: none;
                transition: all 0.1s ease;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <video id="webcam" autoplay playsinline></video>
            <img id="overlay" src="" style="display: none;" />
        </div>

        <script>
            const video = document.getElementById('webcam');
            const overlay = document.getElementById('overlay');

            const outfits = {{
                "Outfit 1": "data:image/png;base64,{outfit1_b64}",
                "Outfit 2": "data:image/png;base64,{outfit2_b64}",
                "Outfit 3": "data:image/png;base64,{outfit3_b64}",
                "None": ""
            }};

            const currentOutfit = "{outfit_choice}";

            if (outfits[currentOutfit] && currentOutfit !== "None") {{
                overlay.src = outfits[currentOutfit];
                overlay.style.display = "block";
            }} else {{
                overlay.style.display = "none";
            }}

            navigator.mediaDevices.getUserMedia({{ video: {{ width: 1280, height: 720 }} }})
                .then(stream => {{
                    video.srcObject = stream;
                }})
                .catch(err => {{
                    console.error("Camera access error: ", err);
                }});
        </script>
    </body>
    </html>
    """
    
    components.html(html_code, height=520)
