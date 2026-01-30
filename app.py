import streamlit as st
import pandas as pd
import os
import requests
import base64
import time
import zipfile
from io import BytesIO
from PIL import Image

# Register AVIF/JFIF plugins explicitly
import pillow_avif

# 1. CONFIG & SETUP
st.set_page_config(
    page_title="Davinci - High-Quality Image Processor",
    page_icon="🎨",
    layout="wide"
)

SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg", "webp", "avif", "jfif"]

# 2. HELPER FUNCTIONS
def get_base64_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return base64.b64encode(response.content).decode('utf-8')
    except:
        return None

def get_base64_from_file(uploaded_file):
    try:
        uploaded_file.seek(0)
        bytes_data = uploaded_file.getvalue()
        return base64.b64encode(bytes_data).decode('utf-8')
    except:
        return None

def resize_with_padding(img, target_size, background_color=(255, 255, 255)):
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P': img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    ratio = min(target_size[0] / img.width, target_size[1] / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    final_img = Image.new("RGB", target_size, background_color)
    paste_pos = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
    final_img.paste(img, paste_pos)
    return final_img

def call_gemini_api(api_key, base64_image, prompt):
    """Optional Gemini API call"""
    if not api_key: return None
    from openai import OpenAI
    BASE_URL = "https://litellm.dhhmena.com/"
    try:
        client = OpenAI(api_key=api_key, base_url=BASE_URL)
        response = client.chat.completions.create(
            model="gemini-3-pro-image-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }],
        )
        resp_dict = response.model_dump()
        message = resp_dict['choices'][0]['message']
        if 'images' in message and message['images']:
            image_obj = message['images'][0]
            if 'image_url' in image_obj and isinstance(image_obj['image_url'], dict):
                return image_obj['image_url'].get('url')
            elif 'url' in image_obj:
                return image_obj['url']
            elif 'b64_json' in image_obj:
                return "data:image/jpeg;base64," + image_obj['b64_json']
        elif message.get('content'):
            import re
            url_match = re.search(r'(https?://[^\s)]+)', message['content'])
            if url_match: return url_match.group(1).strip(')"')
    except Exception as e:
        st.warning(f"Gemini API call failed: {e}")
        return None

def process_image_logic(image_source, source_type, api_key, target_size):
    """Download, optional API call, then resize locally"""
    if source_type == "url":
        base64_img = get_base64_from_url(image_source)
    else:
        base64_img = get_base64_from_file(image_source)
    if not base64_img:
        st.warning("Failed to read image data.")
        return None

    user_prompt = """
    Enhance this product image to high quality (4K).
    Isolate the product on solid white background.
    Fix blur and upscale resolution.
    Preserve product text and logos.
    """

    result_url = call_gemini_api(api_key, base64_img, user_prompt)
    
    # If API fails, fallback to local resize
    try:
        if result_url:
            if result_url.startswith("data:"):
                img_bytes = base64.b64decode(result_url.split(",", 1)[1])
                img = Image.open(BytesIO(img_bytes))
            else:
                resp = requests.get(result_url, timeout=10)
                img = Image.open(BytesIO(resp.content))
        else:
            # fallback to original image
            if source_type == "url":
                resp = requests.get(image_source, timeout=10)
                img = Image.open(BytesIO(resp.content))
            else:
                image_source.seek(0)
                img = Image.open(image_source)
    except:
        st.warning("Failed to open image, using original file.")
        return None

    final_img = resize_with_padding(img, (target_size, target_size))
    return final_img

# 3. STREAMLIT UI
st.title("Davinci - High-Quality Image Processor 🎨")
st.sidebar.header("⚙️ Configuration")

api_key = st.sidebar.text_input("API Key (optional)", type="password")
target_size = st.sidebar.slider("Target Size (px)", 500, 4000, 1400)
processing_mode = st.sidebar.radio("Mode", ["Single Preview", "Batch (Excel/CSV)", "Batch (Local Files)"])
st.sidebar.info("Supported formats: PNG, JPG, WEBP, AVIF, JFIF")

# ---------------- Single Preview ----------------
if processing_mode == "Single Preview":
    input_type = st.radio("Input Source", ["Upload", "URL"], horizontal=True)
    source = None
    img_preview = None

    if input_type == "Upload":
        source = st.file_uploader("Choose image", type=SUPPORTED_EXTENSIONS)
    else:
        url_input = st.text_input("Enter Image URL")
        if url_input: source = url_input

    if source:
        try:
            if input_type == "Upload": img_preview = Image.open(source)
            else: img_preview = Image.open(BytesIO(requests.get(source).content))
        except: img_preview = None

        col1, col2 = st.columns(2)
        with col1: st.image(img_preview, caption="Original", use_container_width=True)
        with col2:
            if st.button("✨ Process Image"):
                with st.spinner("Processing..."):
                    processed = process_image_logic(source, "url" if input_type=="URL" else "file", api_key, target_size)
                    if processed:
                        st.image(processed, caption="Processed", use_container_width=True)
                        buf = BytesIO()
                        processed.save(buf, format="JPEG", quality=95)
                        st.download_button("Download", buf.getvalue(), "enhanced.jpg", "image/jpeg")
                    else:
                        st.error("Failed to process image.")

# ---------------- Batch Excel/CSV ----------------
elif processing_mode == "Batch (Excel/CSV)":
    uploaded_file = st.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            image_col = st.selectbox("Select Image URL Column", df.columns)
            default_idx = 0
            for i, col in enumerate(df.columns):
                if "barcode" in col.lower() or "sku" in col.lower() or "id" in col.lower():
                    default_idx = i
                    break
            barcode_col = st.selectbox("Select Filename Column", df.columns, index=default_idx)

            if st.button("Start Batch Processing"):
                progress_bar = st.progress(0)
                results = []
                total = len(df)
                for idx, row in df.iterrows():
                    progress_bar.progress((idx+1)/total)
                    url = str(row[image_col])
                    barcode = str(row[barcode_col])
                    if pd.isna(url) or url.lower() == "nan" or url=="": continue
                    img = process_image_logic(url, "url", api_key, target_size)
                    if img:
                        buf = BytesIO()
                        img.save(buf, format="JPEG", quality=95)
                        results.append({"filename": f"{barcode}.jpg", "data": buf.getvalue()})
                    time.sleep(0.5)

                # ZIP download
                zip_buf = BytesIO()
                with zipfile.ZipFile(zip_buf, "w") as zf:
                    for item in results: zf.writestr(item["filename"], item["data"])
                st.download_button("Download All as ZIP", zip_buf.getvalue(), "batch_images.zip", "application/zip")
                st.success(f"Processed {len(results)} images.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ---------------- Batch Local Files ----------------
elif processing_mode == "Batch (Local Files)":
    uploaded_files = st.file_uploader("Upload images", type=SUPPORTED_EXTENSIONS, accept_multiple_files=True)
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} images.")
        if st.button("Start Processing Files"):
            progress_bar = st.progress(0)
            results = []
            total = len(uploaded_files)
            for idx, file_obj in enumerate(uploaded_files):
                progress_bar.progress((idx+1)/total)
                base_name = os.path.splitext(file_obj.name)[0]
                img = process_image_logic(file_obj, "file", api_key, target_size)
                if img:
                    buf = BytesIO()
                    img.save(buf, format="JPEG", quality=95)
                    results.append({"filename": f"{base_name}.jpg", "data": buf.getvalue()})
                time.sleep(0.2)
            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for item in results: zf.writestr(item["filename"], item["data"])
            st.download_button("Download All as ZIP", zip_buf.getvalue(), "batch_local.zip", "application/zip")
            st.success(f"Processed {len(results)} images.")
