# app.py

import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from scipy import ndimage
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import cv2
import warnings

# --- Suppress warnings ---
warnings.filterwarnings("ignore")

# --- Page Config ---
st.set_page_config(page_title="Davinci - Image Processor", page_icon="🎨", layout="wide")
st.title("🎨 Davinci - Pro Vivid Edition")

# --- Model Setup (BiRefNet-Lite) ---
@st.cache_resource
def get_birefnet_model():
    """Loads the BiRefNet-Lite model (approx 170MB)."""
    try:
        model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet_lite", 
            trust_remote_code=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Failed to load AI Model: {e}")
        return None, None

# --- 1. AI Background Removal ---
def process_birefnet(img: Image.Image, model, device) -> Image.Image:
    w, h = img.size
    
    # Resize for inference (BiRefNet likes 1024x1024)
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
    
    pred_mask = preds[0].squeeze()
    mask_pil = transforms.ToPILImage()(pred_mask)
    mask_pil = mask_pil.resize((w, h), Image.LANCZOS)
    
    img.putalpha(mask_pil)
    return img

# --- 2. The "Gray Halo" Fix (Despilling) ---
def remove_gray_halo(img_rgba: Image.Image) -> Image.Image:
    """
    Fixes dirty edges by forcing semi-transparent pixels to match 
    the object's color instead of being black/gray.
    """
    # Convert to numpy
    img_np = np.array(img_rgba)
    
    # Extract Alpha
    alpha = img_np[:, :, 3]
    
    # Identify the "Edge Zone" (pixels that are semi-transparent)
    # We define "Edge" as alpha between 0 and 250
    edge_mask = (alpha > 0) & (alpha < 250)
    
    # Identify the "Solid Object" (alpha > 250)
    solid_mask = alpha >= 250
    
    # If we have no solid object, skip
    if not np.any(solid_mask):
        return img_rgba

    # --- COLOR BLEED FIX ---
    # We replace the RGB color of the semi-transparent edges 
    # with the color of the nearest "solid" pixel.
    # This stops the edge from looking "gray" when fading out.
    
    # Create a "Color Source" mask (solid pixels)
    # We dialate the solid color outwards to cover the edges
    for c in range(3): # R, G, B channels
        channel = img_np[:, :, c]
        
        # Replace edge colors with dilated solid colors
        # This mimics "spreading" the object color into the transparency
        # preventing the dark halo effect.
        solid_color_only = np.where(solid_mask, channel, 0)
        
        # Use max filter to spread the bright color outwards
        # Size 5 covers about 2-3 pixels of edge
        spread_color = ndimage.maximum_filter(solid_color_only, size=5)
        
        # Apply this bright color ONLY to the edge pixels
        img_np[:, :, c] = np.where(edge_mask, spread_color, channel)

    return Image.fromarray(img_np)

# --- 3. The "iPhone Vivid" Look (Sharpen + Color) ---
def apply_vivid_enhance(img: Image.Image) -> Image.Image:
    """
    Mimics iPhone post-processing:
    1. Smart Sharpening (Unsharp Mask)
    2. Color/Contrast Boost
    """
    # Convert to OpenCV format for sharpening
    img_cv = np.array(img)
    
    # A. Smart Sharpening (Unsharp Mask)
    # This makes blurry images look crisp without adding noise
    gaussian = cv2.GaussianBlur(img_cv, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(img_cv, 1.5, gaussian, -0.5, 0)
    
    # Convert back to PIL
    img_pil = Image.fromarray(unsharp_image)
    
    # B. Vivid Color Boost (Saturation)
    enhancer = ImageEnhance.Color(img_pil)
    img_pil = enhancer.enhance(1.3) # 30% more color saturation
    
    # C. Contrast Boost (Pop)
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(1.15) # 15% more contrast
    
    return img_pil

# --- Main Pipeline ---
def load_and_orient_image(image_source) -> Image.Image:
    img = Image.open(image_source)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")

def process_final_image(img: Image.Image, target_size: int, padding: int, output_format: str, quality: int, enhance_mode: bool):
    
    # 1. Remove Background
    model, device = get_birefnet_model()
    if model:
        # Remove BG
        img_transparent = process_birefnet(img, model, device)
        
        # 2. Fix Gray Halos (The "Dirty Edge" Fix)
        img_clean = remove_gray_halo(img_transparent)
    else:
        img_clean = img.convert("RGBA")

    # 3. Crop to content
    bbox = img_clean.getbbox()
    if bbox:
        img_clean = img_clean.crop(bbox)

    # 4. Resize Logic
    max_box = target_size - 2 * padding
    width, height = img_clean.size
    ratio = min(max_box / width, max_box / height)
    new_size = (int(width * ratio), int(height * ratio))
    img_resized = img_clean.resize(new_size, Image.LANCZOS)
    
    # 5. Apply "iPhone Vivid" Look (If enabled)
    # We do this AFTER resize so we sharpen the final pixels
    if enhance_mode:
        img_resized = apply_vivid_enhance(img_resized)

    # 6. Place on Canvas
    final_bg_color = (255, 255, 255, 255) if output_format == "JPEG" else (0, 0, 0, 0)
    final_mode = "RGB" if output_format == "JPEG" else "RGBA"
    
    final_img = Image.new(final_mode, (target_size, target_size), final_bg_color)
    
    x_pos = (target_size - new_size[0]) // 2
    y_pos = (target_size - new_size[1]) // 2
    y_pos -= int(target_size * 0.02) # Optical center

    # Paste
    if output_format == "JPEG":
        bg = Image.new("RGB", new_size, (255, 255, 255))
        # Use alpha as mask
        if img_resized.mode == 'RGBA':
            bg.paste(img_resized, mask=img_resized.split()[3])
        else:
            bg.paste(img_resized)
        final_img.paste(bg, (x_pos, y_pos))
    else:
        final_img.paste(img_resized, (x_pos, y_pos), img_resized)
        
    return final_img

# --- UI Setup ---
with st.sidebar:
    st.header("⚙️ Settings")
    processing_mode = st.radio("Mode", ["Single Preview", "Batch (Excel)", "Batch (Local Files)"])
    
    st.subheader("Enhancement")
    # Default to True as requested ("Vivid output like iPhone")
    enhance_mode = st.toggle("✨ Vivid Mode (iPhone Style)", value=True)
    
    st.subheader("Output")
    output_format = st.selectbox("Format", ["JPEG", "PNG"], index=0)
    quality = st.slider("Quality", 70, 100, 95)
    target_size = st.number_input("Canvas Size", value=1400, step=100)
    padding = st.number_input("Padding", value=100, step=10)

# --- Main App Logic ---
if processing_mode == "Single Preview":
    st.subheader("Single Image Preview")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "webp"])
    
    if uploaded_file:
        original = load_and_orient_image(uploaded_file)
        col1, col2 = st.columns(2)
        col1.image(original, caption="Original", use_container_width=True)
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                processed = process_final_image(original, target_size, padding, output_format, quality, enhance_mode)
            col2.image(processed, caption="Processed (Vivid)", use_container_width=True)
            
            buf = BytesIO()
            save_fmt = "JPEG" if output_format == "JPEG" else "PNG"
            processed.save(buf, format=save_fmt, quality=quality)
            st.download_button("Download", buf.getvalue(), file_name=f"vivid_output.{save_fmt.lower()}")

elif processing_mode == "Batch (Excel)":
    st.subheader("Batch Processing (Excel)")
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.dataframe(df.head(3))
        url_col = st.selectbox("URL Column", df.columns)
        name_col = st.selectbox("Filename Column", df.columns)
        
        if st.button("Run Batch"):
            import zipfile
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                progress_bar = st.progress(0)
                for i, row in df.iterrows():
                    try:
                        url = row[url_col]
                        fname = str(row[name_col])
                        resp = requests.get(url, timeout=10)
                        img = load_and_orient_image(BytesIO(resp.content))
                        final = process_final_image(img, target_size, padding, output_format, quality, enhance_mode)
                        
                        img_byte_arr = BytesIO()
                        save_fmt = "JPEG" if output_format == "JPEG" else "PNG"
                        final.save(img_byte_arr, format=save_fmt, quality=quality)
                        zf.writestr(f"{fname}.{save_fmt.lower()}", img_byte_arr.getvalue())
                    except Exception as e:
                        pass
                    progress_bar.progress((i + 1) / len(df))
            st.download_button("Download Batch ZIP", zip_buffer.getvalue(), file_name="batch.zip")

elif processing_mode == "Batch (Local Files)":
    st.subheader("Batch (Local Files)")
    uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)
    if uploaded_files and st.button("Run Batch"):
        import zipfile
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            progress_bar = st.progress(0)
            for i, f in enumerate(uploaded_files):
                try:
                    img = load_and_orient_image(f)
                    final = process_final_image(img, target_size, padding, output_format, quality, enhance_mode)
                    
                    img_byte_arr = BytesIO()
                    save_fmt = "JPEG" if output_format == "JPEG" else "PNG"
                    final.save(img_byte_arr, format=save_fmt, quality=quality)
                    zf.writestr(f"{f.name.rsplit('.',1)[0]}.{save_fmt.lower()}", img_byte_arr.getvalue())
                except: pass
                progress_bar.progress((i + 1) / len(uploaded_files))
        st.download_button("Download Batch ZIP", zip_buffer.getvalue(), file_name="batch.zip")
