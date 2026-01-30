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

# Register AVIF/JFIF plugins explicitly if needed
# usually 'pillow-avif-plugin' auto-registers on import
import pillow_avif

# 1. CONFIG & SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Davinci - High-Quality Image Processor", 
    page_icon="🎨", 
    layout="wide"
)

# Default constants
DEFAULT_API_KEY = "sk-dfMwzaOYxgMv2m_eesW-tw"
BASE_URL = "https://litellm.dhhmena.com/"
# Extended file support
SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg", "webp", "avif", "jfif"]

# 2. HELPER FUNCTIONS
# ---------------------------------------------------------
def get_base64_from_url(url):
    """Downloads image from URL and converts to Base64"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return base64.b64encode(response.content).decode('utf-8')
    except Exception as e:
        # st.error(f"Error downloading URL: {e}") 
        return None

def get_base64_from_file(uploaded_file):
    """Converts uploaded Streamlit file object to Base64"""
    try:
        # Reset pointer to start just in case
        uploaded_file.seek(0)
        bytes_data = uploaded_file.getvalue()
        return base64.b64encode(bytes_data).decode('utf-8')
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def resize_with_padding(img, target_size, background_color=(255, 255, 255)):
    """Resizes image to fit target_size, padding the rest with white."""
    # Convert RGBA/LA/P to RGB (white bg)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    ratio = min(target_size[0] / img.width, target_size[1] / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    final_img = Image.new("RGB", target_size, background_color)
    paste_pos = ((target_size[0] - new_size[0]) // 2, 
                 (target_size[1] - new_size[1]) // 2)
    final_img.paste(img, paste_pos)
    return final_img

def call_gemini_api(api_key, base64_image, prompt):
    """Calls the Gemini API via OpenAI client compatibility"""
    if not api_key:
        st.error("Missing API Key.")
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

        # Parse Response
        response_dict = response.model_dump()
        message = response_dict['choices'][0]['message']
        final_image_url = None

        if 'images' in message and message['images']:
            image_obj = message['images'][0]
            if 'image_url' in image_obj and isinstance(image_obj['image_url'], dict):
                final_image_url = image_obj['image_url'].get('url')
            elif 'url' in image_obj:
                final_image_url = image_obj['url']
            elif 'b64_json' in image_obj:
                final_image_url = "data:image/jpeg;base64," + image_obj['b64_json']
        elif message.get('content'):
            url_match = re.search(r'(https?://[^\s)]+)', message['content'])
            if url_match:
                final_image_url = url_match.group(1).strip(')"')
        
        return final_image_url

    except Exception as e:
        print(f"API Error: {e}")
        return None

def process_image_logic(image_source, source_type, api_key, target_size):
    """Orchestrates the download, API call, and resizing"""
    
    # 1. Get Base64
    if source_type == "url":
        base64_img = get_base64_from_url(image_source)
    else:
        base64_img = get_base64_from_file(image_source)
    
    if not base64_img:
        return None

    # 2. Define Prompt
    user_prompt = """
    Enhance this product image to high quality (4K).
    1. Isolate the product on a SOLID WHITE background (hex #FFFFFF).
    2. Fix any blur and upscale the resolution.
    3. Preserve the product text and logos exactly.
    """

    # 3. Call API
    result_url = call_gemini_api(api_key, base64_img, user_prompt)
    
    if not result_url:
        return None

    # 4. Download Result & Resize
    try:
        img = None
        if result_url.startswith("data:"):
            base64_str = result_url.split(",", 1)[1]
            img_bytes = base64.b64decode(base64_str)
            img = Image.open(BytesIO(img_bytes))
        elif result_url.startswith("http"):
            img_response = requests.get(result_url)
            img = Image.open(BytesIO(img_response.content))
        
        if img:
            final_img = resize_with_padding(img, (target_size, target_size))
            return final_img
    except Exception as e:
        print(f"Error processing API result: {e}")
        return None

# 3. UI LAYOUT
# ---------------------------------------------------------
st.title("Davinci - High-Quality Image Processor 🎨")
st.markdown("Upscale, Fix Blur, and Isolate Backgrounds using Gemini-3-Pro.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input("API Key", value=DEFAULT_API_KEY, type="password")
    
    st.subheader("Output Settings")
    target_size = st.slider("Target Size (px)", 500, 4000, 1400)
    
    processing_mode = st.radio(
        "Choose Mode", 
        ["Single Preview", "Batch (Excel/CSV)", "Batch (Local Files)"]
    )
    
    st.info("Supported formats: PNG, JPG, WEBP, AVIF, JFIF")

# ---------------------------------------------------------
# MODE 1: SINGLE PREVIEW
# ---------------------------------------------------------
if processing_mode == "Single Preview":
    st.header("Single Image Preview")
    
    input_type = st.radio("Input Source", ["Upload", "URL"], horizontal=True)
    
    source = None
    original_image_preview = None
    
    if input_type == "Upload":
        source = st.file_uploader("Choose an image", type=SUPPORTED_EXTENSIONS)
        if source:
            try:
                original_image_preview = Image.open(source)
            except Exception as e:
                st.error(f"Error opening image: {e}")
    else:
        url_input = st.text_input("Enter Image URL")
        if url_input:
            source = url_input
            try:
                resp = requests.get(source, timeout=5)
                original_image_preview = Image.open(BytesIO(resp.content))
            except:
                st.warning("Could not load preview from URL.")

    if source:
        col1, col2 = st.columns(2)
        with col1:
            if original_image_preview:
                st.image(original_image_preview, caption="Original", use_container_width=True)
            else:
                st.write("Original Image")

        with col2:
            if st.button("✨ Process Image", type="primary"):
                with st.spinner("Processing... (10-20s)"):
                    processed_img = process_image_logic(
                        image_source=source, 
                        source_type="url" if input_type == "URL" else "file",
                        api_key=api_key,
                        target_size=target_size
                    )
                    
                    if processed_img:
                        st.image(processed_img, caption="Processed Result", use_container_width=True)
                        
                        # Download Button
                        buf = BytesIO()
                        processed_img.save(buf, format="JPEG", quality=95)
                        st.download_button(
                            label="Download Result",
                            data=buf.getvalue(),
                            file_name="enhanced_image.jpg",
                            mime="image/jpeg"
                        )
                    else:
                        st.error("Failed to process image.")

# ---------------------------------------------------------
# MODE 2: BATCH (EXCEL / CSV)
# ---------------------------------------------------------
elif processing_mode == "Batch (Excel/CSV)":
    st.header("Batch Process from File (URL List)")
    
    uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])
    
    if uploaded_file:
        try:
            # Determine loader based on extension
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, dtype=str)
            else:
                df = pd.read_excel(uploaded_file, dtype=str)
                
            st.dataframe(df.head())
            
            col1, col2 = st.columns(2)
            with col1:
                image_col = st.selectbox("Select Image URL Column", df.columns)
            with col2:
                # Try to find a 'Barcode' or 'SKU' column automatically
                default_idx = 0
                for i, col in enumerate(df.columns):
                    if "barcode" in col.lower() or "sku" in col.lower() or "id" in col.lower():
                        default_idx = i
                        break
                barcode_col = st.selectbox("Select Filename/Barcode Column", df.columns, index=default_idx)
                
            st.caption(f"Found {len(df)} rows to process.")
            
            # Session State for results
            if 'batch_results_excel' not in st.session_state:
                st.session_state.batch_results_excel = []
            
            if st.button("Start Batch Processing"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                total_rows = len(df)
                
                for index, row in df.iterrows():
                    # Progress Update
                    progress = (index + 1) / total_rows
                    progress_bar.progress(progress)
                    
                    barcode = str(row[barcode_col]).strip()
                    url = str(row[image_col]).strip()
                    
                    status_text.text(f"Processing {index+1}/{total_rows}: {barcode}...")
                    
                    if pd.isna(url) or url.lower() == "nan" or url == "":
                        continue
                        
                    processed_img = process_image_logic(url, "url", api_key, target_size)
                    
                    if processed_img:
                        buf = BytesIO()
                        processed_img.save(buf, format="JPEG", quality=95)
                        results.append({
                            "filename": f"{barcode}.jpg",
                            "data": buf.getvalue()
                        })
                    
                    # Rate limit safety
                    time.sleep(1)
                
                st.session_state.batch_results_excel = results
                status_text.text("Processing Complete!")
                st.success(f"Successfully processed {len(results)} images.")

            # Download Logic
            if st.session_state.batch_results_excel:
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for item in st.session_state.batch_results_excel:
                        zf.writestr(item["filename"], item["data"])
                
                st.download_button(
                    label="Download All as ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="enhanced_images_urls.zip",
                    mime="application/zip",
                    type="primary"
                )

        except Exception as e:
            st.error(f"Error reading file: {e}")

# ---------------------------------------------------------
# MODE 3: BATCH (LOCAL FILES)
# ---------------------------------------------------------
elif processing_mode == "Batch (Local Files)":
    st.header("Batch Process Local Files")
    st.markdown("Upload multiple images directly from your computer.")
    
    uploaded_files = st.file_uploader(
        "Upload images", 
        type=SUPPORTED_EXTENSIONS, 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} images.")
        
        # Session State for results
        if 'batch_results_local' not in st.session_state:
            st.session_state.batch_results_local = []
        
        if st.button("Start Processing Files"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            total_files = len(uploaded_files)
            
            for index, file_obj in enumerate(uploaded_files):
                # Progress
                progress = (index + 1) / total_files
                progress_bar.progress(progress)
                
                # Get filename without extension for saving
                base_name = os.path.splitext(file_obj.name)[0]
                status_text.text(f"Processing {index+1}/{total_files}: {file_obj.name}...")
                
                # Process
                processed_img = process_image_logic(file_obj, "file", api_key, target_size)
                
                if processed_img:
                    buf = BytesIO()
                    processed_img.save(buf, format="JPEG", quality=95)
                    results.append({
                        "filename": f"{base_name}.jpg",
                        "data": buf.getvalue()
                    })
                
                time.sleep(0.5) # Slight delay
            
            st.session_state.batch_results_local = results
            status_text.text("Processing Complete!")
            st.success(f"Successfully processed {len(results)} images.")
            
        # Download Logic
        if st.session_state.batch_results_local:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for item in st.session_state.batch_results_local:
                    zf.writestr(item["filename"], item["data"])
            
            st.download_button(
                label="Download All as ZIP",
                data=zip_buffer.getvalue(),
                file_name="enhanced_local_images.zip",
                mime="application/zip",
                type="primary"
            )
