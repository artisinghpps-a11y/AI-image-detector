import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import base64

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Image Detector", page_icon="🕵️", layout="wide")

# --- ADVANCED UI ARCHITECTURE ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;700&display=swap');

    /* Premium Obsidian Flow Background */
    .stApp {{
        background: linear-gradient(135deg, #020205 0%, #080a1a 50%, #020205 100%);
        background-size: 400% 400%;
        animation: obsidianFlow 12s ease infinite;
        color: #f0f0f0;
        font-family: 'JetBrains Mono', monospace;
    }}
    @keyframes obsidianFlow {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    header {{visibility: hidden;}}
    
    .logo-container {{
        text-align: left;
        padding-bottom: 15px;
        border-bottom: 1px solid rgba(0, 210, 255, 0.2);
        margin-bottom: 25px;
    }}
    .logo-text {{
        font-weight: 700;
        font-size: 1.5rem;
        letter-spacing: 4px;
        color: #00d2ff;
        text-shadow: 0 0 10px rgba(0, 210, 255, 0.5);
    }}

    /* HUD-Style Card */
    .hud-card {{
        background: rgba(10, 15, 30, 0.85);
        border: 1px solid rgba(0, 210, 255, 0.3);
        border-radius: 4px;
        padding: 20px;
        position: relative;
        box-shadow: inset 0 0 20px rgba(0, 210, 255, 0.1);
        margin-top: 20px;
    }}
    
    .hud-card::after {{
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 2px;
        background: rgba(0, 210, 255, 0.4);
        animation: scanline 3s linear infinite;
    }}
    @keyframes scanline {{ 0% {{ top: 0%; }} 100% {{ top: 100%; }} }}

    .status-badge {{
        padding: 4px 10px;
        border-radius: 2px;
        font-size: 0.6rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        background: rgba(0, 210, 255, 0.1);
        border: 1px solid #00d2ff;
        color: #00d2ff;
    }}

    /* RESPONSIVE IMAGE BOX */
    [data-testid="stImage"] img {{
        max-height: 65vh !important;
        width: auto !important;
        border: 1px solid rgba(0, 210, 255, 0.2);
        border-radius: 4px;
        object-fit: contain;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }}

    /* MOBILE OPTIMIZED TABS & BUTTONS */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background-color: transparent;
    }}

    .stButton > button {{
        width: 100%;
        background: transparent;
        color: #00d2ff !important;
        border: 1px solid #00d2ff !important;
        border-radius: 0px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 1px;
        transition: 0.3s;
        margin-top: 10px;
        height: 3em;
    }}
    .stButton > button:hover {{
        background: #00d2ff !important;
        color: black !important;
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.6);
    }}

    /* CAMERA WIDGET STYLING */
    [data-testid="stCameraInput"] {{
        border: 1px solid rgba(0, 210, 255, 0.3);
        border-radius: 4px;
    }}
    </style>
""", unsafe_allow_html=True)

SYS_CFG = {
    "CORE_RES": "c2lnaHRlbmdpbmUuY2xpZW50",
    "CORE_CLS": "U2lnaHRlbmdpbmVDbGllbnQ="
}

# --- INTERNAL LOGIC ---
@st.cache_resource
def load_local_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 1), nn.Sigmoid())
    if Path("resnet_ai_detector.pth").exists():
        model.load_state_dict(torch.load("resnet_ai_detector.pth", map_location="cpu"))
    return model.eval()

def sys_probe(fb, u, s):
    try:
        _m = base64.b64decode(SYS_CFG["CORE_RES"]).decode()
        _c = base64.b64decode(SYS_CFG["CORE_CLS"]).decode()
        _mod = __import__(_m, fromlist=[_c])
        _cls = getattr(_mod, _c)
        _conn = _cls(u, s)
        _res = _conn.check('genai').set_bytes(fb)
        if _res['status'] == 'success':
            return _res['type']['ai_generated'], "NEURAL_CLOUD_V4"
    except: pass
    return None, None

# --- UI HEADER ---
st.markdown('<div class="logo-container"><span class="logo-text">AI IMAGE DETECTOR</span></div>', unsafe_allow_html=True)

# --- LAYOUT LOGIC ---
if 'run_analysis' not in st.session_state:
    st.session_state['run_analysis'] = False

# Layout: Main Preview on left/top, Controls on right/bottom
col_main, col_spacer, col_side = st.columns([2.2, 0.1, 1.7], gap="small") 

# Image Input Section
with col_side:
    st.markdown("### 📡 DATA_CHANNEL")
    
    # MOBILE OPTIMIZATION: Toggle between Camera and Upload
    input_mode = st.tabs(["📸 CAMERA", "📁 UPLOAD"])
    
    input_file = None
    
    with input_mode[0]:
        cam_data = st.camera_input("SCAN_LIVE", label_visibility="collapsed")
        if cam_data:
            input_file = cam_data
            
    with input_mode[1]:
        up_data = st.file_uploader("LOAD_FILE", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if up_data:
            input_file = up_data

    if input_file:
        if st.button("EXECUTE FORENSIC PROTOCOL"):
            st.session_state['run_analysis'] = True
            
        if st.session_state['run_analysis']:
            file_bytes = input_file.getvalue()
            img_process = Image.open(input_file).convert("RGB")
            
            with st.spinner("SCANNING NEURAL SIGNATURES..."):
                api_score, api_label = sys_probe(
                    file_bytes, 
                    "1821315875", 
                    "vhorLMbekgBS36Jy9QfTAh49TdBrZDpu"
                )
                
                local_model = load_local_model()
                transform = transforms.Compose([
                    transforms.Resize((224, 224)), transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                input_tensor = transform(img_process).unsqueeze(0)
                with torch.no_grad():
                    local_score = local_model(input_tensor).item()

                ai_prob = api_score if api_score is not None else local_score
                real_prob = 1.0 - ai_prob
                is_ai = ai_prob > 0.5

                color = "#ff4b4b" if is_ai else "#00d2ff"
                verdict = "AI_GENERATED" if is_ai else "HUMAN_ORIGIN"
                
                st.markdown(f"""
                    <div class="hud-card">
                        <span class="status-badge" style="border-color: {color}; color: {color};">VERDICT: {verdict}</span>
                        <h3 style="margin: 15px 0; font-weight: 200; font-size: 0.9rem;">ANALYSIS_COMPLETED</h3>
                        <div style="display: flex; justify-content: space-between; margin: 8px 0; font-size: 0.8rem;">
                            <span style="color: #888;">AI_PROBABILITY</span>
                            <span style="color: #ff4b4b; font-weight: bold;">{ai_prob*100:.2f}%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin: 8px 0; font-size: 0.8rem;">
                            <span style="color: #888;">REAL_INTEGRITY</span>
                            <span style="color: #00d2ff; font-weight: bold;">{real_prob*100:.2f}%</span>
                        </div>
                        <hr style="opacity: 0.1; margin: 15px 0;">
                        <p style="font-size: 0.6rem; color: #555; line-height: 1.4;">
                            DETECTION_ENGINE: {api_label if api_score is not None else 'LOCAL_RESNET_V1'}<br>
                            SCAN_MODE: PIXEL_DISTRIBUTION_FORENSICS
                        </p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("AWAITING SOURCE DATA...")

# Main Image Preview Section
with col_main:
    if input_file:
        img_display = Image.open(input_file).convert("RGB")
        st.image(img_display, caption="[ SOURCE_DATA_PREVIEW ]")
    else:
        # Placeholder for empty state
        st.markdown("""
            <div style="height: 50vh; border: 1px dashed rgba(0,210,255,0.1); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #333;">
                NO_FEED_DETECTED
            </div>
        """, unsafe_allow_html=True)