import streamlit as st
import pandas as pd
import os
import requests
import base64
import time
import zipfile
from io import BytesIO
from PIL import Image

# AVIF/JFIF support
import pillow_avif

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Davinci - High-Quality Image Processor", layout="wide")
SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg", "webp", "avif", "jfif"]

# ---------------- HELPERS ----------------
def resize_with_padding(img, target_size, bg_color=(255, 255, 255)):
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        background = Image.new('RGB', img.size, bg_color)
        if img.mode == 'P': img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    ratio = min(target_size[0]/img.width, target_size[1]/img.height)
    new_size = (int(img.width*ratio), int(img.height*ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    final_img = Image.new('RGB', target_size, bg_color)
    final_img.paste(img, ((target_size[0]-new_size[0])//2, (target_size[1]-new_size[1])//2))
    return final_img

def process_image(image_source, source_type, api_key, target_size):
    # Try to open image from file/url
    try:
        if source_type == 'file':
            image_source.seek(0)
            img = Image.open(image_source)
        else:
            resp = requests.get(image_source, timeout=10)
            img = Image.open(BytesIO(resp.content))
    except:
        return None
    return resize_with_padding(img, (target_size, target_size))

# ---------------- UI ----------------
st.title("Davinci - High-Quality Image Processor 🎨")
st.sidebar.header("⚙️ Settings")
api_key = st.sidebar.text_input("API Key (optional)", type="password")
target_size = st.sidebar.slider("Target Size (px)", 500, 4000, 1400)
mode = st.sidebar.radio("Mode", ["Single Preview", "Batch (Excel/CSV)", "Batch (Local Files)"])
st.sidebar.info("Supported: PNG, JPG, JPEG, WEBP, AVIF, JFIF")

# ---------------- SINGLE ----------------
if mode == "Single Preview":
    source_type = st.radio("Input Source", ["Upload", "URL"], horizontal=True)
    source = None
    preview = None

    if source_type == "Upload":
        source = st.file_uploader("Choose Image", type=SUPPORTED_EXTENSIONS)
    else:
        url = st.text_input("Enter Image URL")
        if url: source = url

    if source:
        try:
            if source_type == "Upload": preview = Image.open(source)
            else: preview = Image.open(BytesIO(requests.get(source).content))
        except: preview = None

        col1, col2 = st.columns(2)
        with col1:
            st.image(preview, caption="Original", use_container_width=True)
        with col2:
            if st.button("✨ Process Image"):
                with st.spinner("Processing..."):
                    processed = process_image(source, "url" if source_type=="URL" else "file", api_key, target_size)
                    if processed:
                        st.image(processed, caption="Processed", use_container_width=True)
                        buf = BytesIO()
                        processed.save(buf, format="JPEG", quality=95)
                        st.download_button("Download", buf.getvalue(), "enhanced.jpg", "image/jpeg")
                    else:
                        st.error("Failed to process image.")

# ---------------- BATCH CSV/EXCEL ----------------
elif mode == "Batch (Excel/CSV)":
    uploaded_file = st.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            image_col = st.selectbox("Select Image URL Column", df.columns)
            barcode_col = st.selectbox("Select Filename Column", df.columns)
            if st.button("Start Batch"):
                results = []
                progress = st.progress(0)
                for i, row in df.iterrows():
                    progress.progress((i+1)/len(df))
                    img = process_image(str(row[image_col]), "url", api_key, target_size)
                    if img:
                        buf = BytesIO()
                        img.save(buf, format="JPEG", quality=95)
                        results.append({"filename": f"{row[barcode_col]}.jpg", "data": buf.getvalue()})
                    time.sleep(0.2)
                # ZIP download
                zip_buf = BytesIO()
                with zipfile.ZipFile(zip_buf, "w") as zf:
                    for item in results: zf.writestr(item["filename"], item["data"])
                st.download_button("Download ZIP", zip_buf.getvalue(), "batch_images.zip", "application/zip")
        except Exception as e:
            st.error(f"Failed: {e}")

# ---------------- BATCH LOCAL ----------------
elif mode == "Batch (Local Files)":
    files = st.file_uploader("Upload Images", type=SUPPORTED_EXTENSIONS, accept_multiple_files=True)
    if files:
        if st.button("Start Batch"):
            results = []
            progress = st.progress(0)
            for i, f in enumerate(files):
                progress.progress((i+1)/len(files))
                img = process_image(f, "file", api_key, target_size)
                if img:
                    buf = BytesIO()
                    img.save(buf, format="JPEG", quality=95)
                    results.append({"filename": f.name, "data": buf.getvalue()})
                time.sleep(0.2)
            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for item in results: zf.writestr(item["filename"], item["data"])
            st.download_button("Download ZIP", zip_buf.getvalue(), "batch_local.zip", "application/zip")
