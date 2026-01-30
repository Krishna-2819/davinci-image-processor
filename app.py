import streamlit as st
import pandas as pd
import os
import requests
import base64
import time
import re
import zipfile
from io import BytesIO
from PIL import Image
from openai import OpenAI
import pillow_avif  # Ensure this is in requirements.txt

# 1. CONFIG & SETUP
st.set_page_config(
    page_title="Davinci - High-Quality Image Processor", 
    page_icon="🎨", 
    layout="wide"
)

# Default constants
BASE_URL = "https://litellm.dhhmena.com/"
SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg", "webp", "avif", "jfif"]

# 2. HELPER FUNCTIONS
def get_base64_from_url(url):
    """Downloads image from URL and converts to Base64"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return base64.b64encode(response.content).decode('utf-8')
    except Exception as e:
        st.error(f"Failed to download image from URL: {e}")
        return None

def get_base64_from_file(uploaded_file):
    """Converts uploaded Streamlit file object to Base64"""
    try:
        uploaded_file.seek(0)
        bytes_data = uploaded_file.getvalue()
        return base64.b64encode(bytes_data).decode('utf-8')
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        return None

def resize_with_padding(img, target_size, background_color=(255, 255, 255)):
    """Resizes image to fit target_size, padding with white."""
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    ratio = min(target_size[0] / img.width, target_size[1] / img.height)
    new_size = (max(1, int(img.width * ratio)), max(1, int(img.height * ratio)))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    final_img = Image.new("RGB", target_size, background_color)
    paste_pos = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
    final_img.paste(img, paste_pos)
    return final_img

def call_gemini_api(api_key, base64_image, prompt, debug=False):
    """Calls the Gemini API via OpenAI client compatibility"""
    if not api_key:
        st.error("Missing API Key. Please enter it in the sidebar.")
        return None

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    
    try:
        response = client.chat.completions.create(
            model="gemini-3-pro-image-preview", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
        )

        # Handle various response formats from LiteLLM/Gemini
        res_content = response.choices[0].message.content
        
        # Check if the response contains an image object (DALL-E style)
        if hasattr(response, 'images') and response.images:
            return response.images[0].url or response.images[0].b64_json
        
        # Fallback: Extract URL from text content using regex
        url_match = re.search(r'(https?://[^\s)]+)', res_content)
        if url_match:
            return url_match.group(1).strip(')"')
        
        if debug:
            st.info(f"Raw API Content: {res_content}")
        return None

    except Exception as e:
        st.error(f"API Call Failed: {e}")
        return None

def process_image_logic(image_source, source_type, api_key, target_size, debug=False):
    base64_img = get_base64_from_url(image_source) if source_type == "url" else get_base64_from_file(image_source)
    
    if not base64_img:
        return None

    user_prompt = "Enhance this product image to 4K high quality. Isolate the product on a SOLID WHITE background (#FFFFFF). Preserve logos."

    result_url = call_gemini_api(api_key, base64_img, user_prompt, debug)
    
    if not result_url:
        return None

    try:
        if result_url.startswith("data:"):
            img_bytes = base64.b64decode(result_url.split(",", 1)[1])
            img = Image.open(BytesIO(img_bytes))
        else:
            img_response = requests.get(result_url, timeout=20)
            img = Image.open(BytesIO(img_response.content))
        
        return resize_with_padding(img, (target_size, target_size))
    except Exception as e:
        st.error(f"Error downloading/processing result image: {e}")
        return None

# 3. UI LAYOUT
st.title("Davinci - High-Quality Image Processor 🎨")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("API Key", type="password")
    target_size = st.slider("Target Size (px)", 500, 4000, 1400)
    debug_mode = st.checkbox("Show Debug Info")
    
    processing_mode = st.radio("Choose Mode", ["Single Preview", "Batch (Excel/CSV)", "Batch (Local Files)"])
    st.info("Supported: PNG, JPG, WEBP, AVIF, JFIF")

# --- MODE 1: SINGLE PREVIEW ---
if processing_mode == "Single Preview":
    st.header("Single Image Preview")
    input_type = st.radio("Input Source", ["Upload", "URL"], horizontal=True)
    
    source = None
    if input_type == "Upload":
        source = st.file_uploader("Choose an image", type=SUPPORTED_EXTENSIONS)
    else:
        source = st.text_input("Enter Image URL")

    if source:
        if st.button("✨ Process Image", type="primary"):
            with st.spinner("Processing via Gemini..."):
                processed_img = process_image_logic(source, input_type.lower(), api_key, target_size, debug_mode)
                if processed_img:
                    st.image(processed_img, caption="Processed Result", use_container_width=True)
                    buf = BytesIO()
                    processed_img.save(buf, format="JPEG", quality=95)
                    st.download_button("Download Result", buf.getvalue(), "enhanced.jpg", "image/jpeg")
                else:
                    st.error("Failed to process image. Check your API key or the debug log.")

# --- MODE 2: BATCH (EXCEL / CSV) ---
elif processing_mode == "Batch (Excel/CSV)":
    st.header("Batch Process from File")
    uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        image_col = col1.selectbox("Image URL Column", df.columns)
        barcode_col = col2.selectbox("Filename/Barcode Column", df.columns)
        
        if st.button("Start Batch Processing"):
            results = []
            progress_bar = st.progress(0)
            for i, (idx, row) in enumerate(df.iterrows()):
                res = process_image_logic(str(row[image_col]), "url", api_key, target_size, debug_mode)
                if res:
                    buf = BytesIO()
                    res.save(buf, format="JPEG")
                    results.append({"filename": f"{row[barcode_col]}.jpg", "data": buf.getvalue()})
                progress_bar.progress((i + 1) / len(df))
            
            if results:
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for item in results:
                        zf.writestr(item["filename"], item["data"])
                st.download_button("Download ZIP", zip_buffer.getvalue(), "batch_results.zip", "application/zip")

# --- MODE 3: BATCH (LOCAL FILES) ---
elif processing_mode == "Batch (Local Files)":
    st.header("Batch Process Local Files")
    uploaded_files = st.file_uploader("Upload images", type=SUPPORTED_EXTENSIONS, accept_multiple_files=True)
    if uploaded_files and st.button("Process All"):
        results = []
        progress_bar = st.progress(0)
        for i, file in enumerate(uploaded_files):
            res = process_image_logic(file, "file", api_key, target_size, debug_mode)
            if res:
                buf = BytesIO()
                res.save(buf, format="JPEG")
                results.append({"filename": f"{os.path.splitext(file.name)[0]}.jpg", "data": buf.getvalue()})
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if results:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for item in results:
                    zf.writestr(item["filename"], item["data"])
            st.download_button("Download ZIP", zip_buffer.getvalue(), "local_batch.zip", "application/zip")
