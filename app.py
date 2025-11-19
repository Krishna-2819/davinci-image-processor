# app.py

import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
from scipy import ndimage
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import warnings

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Page Config ---
st.set_page_config(page_title="Davinci - BiRefNet Edition", page_icon="🎨", layout="wide")
st.title("🎨 Davinci - Pro Background Remover (BiRefNet Lite)")

# --- Model Setup (BiRefNet-Lite) ---
@st.cache_resource
def get_birefnet_model():
    """Loads the BiRefNet-Lite model from Hugging Face."""
    try:
        # Using the 'Lite' version (approx 170MB) to fit in free tier memory
        model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet_lite", 
            trust_remote_code=True
        )
        
        # Set device (Use GPU if available, else CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Failed to load AI Model: {e}")
        return None, None

# --- Image Preprocessing for BiRefNet ---
def process_birefnet(img: Image.Image, model, device) -> Image.Image:
    """Runs the image through the AI model to generate a transparent background."""
    
    # 1. Prepare Input
    # BiRefNet likes 1024x1024 input for best accuracy
    w, h = img.size
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # 2. Run Inference
    with torch.no_grad():
        # 'preds' usually returns a list, we take the last one (final output)
        preds = model(input_tensor)[-1].sigmoid().cpu()
    
    # 3. Process Mask
    pred_mask = preds[0].squeeze()
    mask_pil = transforms.ToPILImage()(pred_mask)
    
    # Resize mask back to original image size
    mask_pil = mask_pil.resize((w, h), Image.LANCZOS)
    
    # 4. Apply Mask to Original Image
    img.putalpha(mask_pil)
    return img

# --- Smart Cleanup (Safe Version) ---
def apply_smart_cleanup(img_rgba: Image.Image) -> Image.Image:
    """Removes gray artifacts without deleting white products."""
    data = np.array(img_rgba)
    
    # Split channels
    r, g, b, alpha = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    
    # 1. Define the "Solid Core" (Pixels that are definitely the object)
    # We lower threshold to 200 to catch semi-transparent edges of white products
    core_mask = alpha > 200
    
    # 2. Create a "Shield" around the core
    # Dilation=6 expands safety zone by ~6px to protect product edges
    protection_zone = ndimage.binary_dilation(core_mask, iterations=6)
    
    # 3. Identify "Gray Ghosts" (The noise you want to remove)
    # We only attack pixels that are VERY faint (alpha < 150) and mostly white/gray
    is_gray_artifact = (
        (alpha > 0) & (alpha < 150) & 
        (r > 220) & (g > 220) & (b > 220) & # Only target very light pixels
        (abs(r.astype(np.int16) - g.astype(np.int16)) < 20) & 
        (abs(g.astype(np.int16) - b.astype(np.int16)) < 20)
    )
    
    # 4. Delete the ghosts ONLY if they are outside the shield
    pixels_to_delete = is_gray_artifact & ~protection_zone
    alpha[pixels_to_delete] = 0
    
    data[:, :, 3] = alpha
    return Image.fromarray(data)

# --- Main Processing Pipeline ---
def load_and_orient_image(image_source) -> Image.Image:
    img = Image.open(image_source)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")

def process_final_image(img: Image.Image, target_size: int, padding: int, output_format: str, quality: int):
    
    # 1. Remove Background (AI)
    model, device = get_birefnet_model()
    if model:
        with st.spinner("AI removing background..."):
            img_transparent = process_birefnet(img, model, device)
            
        # 2. Smart Cleanup
        img_cleaned = apply_smart_cleanup(img_transparent)
        
        # 3. Crop to content (Trimming empty space)
        bbox = img_cleaned.getbbox()
        if bbox:
            img_for_canvas = img_cleaned.crop(bbox)
        else:
            img_for_canvas = img_cleaned # Fallback if empty
    else:
        img_for_canvas = img.convert("RGBA") # Fallback if model fails

    # 4. Resize and Place on Canvas
    max_box = target_size - 2 * padding
    
    # Calculate scaling ratio
    width, height = img_for_canvas.size
    ratio = min(max_box / width, max_box / height)
    new_size = (int(width * ratio), int(height * ratio))
    
    img_resized = img_for_canvas.resize(new_size, Image.LANCZOS)
    
    # Center calculation (with optical offset)
    final_bg_color = (255, 255, 255, 255) if output_format == "JPEG" else (0, 0, 0, 0)
    final_mode = "RGB" if output_format == "JPEG" else "RGBA"
    
    final_img = Image.new(final_mode, (target_size, target_size), final_bg_color)
    
    x_pos = (target_size - new_size[0]) // 2
    y_pos = (target_size - new_size[1]) // 2
    y_pos -= int(target_size * 0.02) # Slight optical shift up (2%)

    # Paste
    if output_format == "JPEG":
        # Composite over white for JPEG
        bg = Image.new("RGB", new_size, (255, 255, 255))
        bg.paste(img_resized, mask=img_resized.split()[3]) # Use alpha as mask
        final_img.paste(bg, (x_pos, y_pos))
    else:
        # Paste transparent for PNG
        final_img.paste(img_resized, (x_pos, y_pos), img_resized)
        
    return final_img

# --- Sidebar UI ---
with st.sidebar:
    st.header("⚙️ Settings")
    processing_mode = st.radio("Mode", ["Single Preview", "Batch (Excel)", "Batch (Local Files)"])
    
    st.info("Using **BiRefNet-Lite** model.\n(Optimized for mixed inventory: Pets, Electronics, Food)")
    
    output_format = st.selectbox("Output Format", ["PNG", "JPEG"])
    quality = st.slider("Quality", 70, 100, 95)
    
    target_size = st.number_input("Canvas Size (px)", value=1400, step=100)
    padding = st.number_input("Padding (px)", value=100, step=10)

# --- Main UI ---

# 1. SINGLE PREVIEW
if processing_mode == "Single Preview":
    st.subheader("Single Image Preview")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "webp"])
    
    if uploaded_file:
        original = load_and_orient_image(uploaded_file)
        col1, col2 = st.columns(2)
        col1.image(original, caption="Original", use_container_width=True)
        
        if st.button("Process Image"):
            processed = process_final_image(original, target_size, padding, output_format, quality)
            col2.image(processed, caption="Processed", use_container_width=True)
            
            # Download Button
            buf = BytesIO()
            save_fmt = "JPEG" if output_format == "JPEG" else "PNG"
            processed.save(buf, format=save_fmt, quality=quality)
            st.download_button("Download Result", buf.getvalue(), file_name=f"processed.{save_fmt.lower()}")

# 2. BATCH (EXCEL)
elif processing_mode == "Batch (Excel)":
    st.subheader("Batch Processing (Excel URLs)")
    uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.dataframe(df.head(3))
        
        url_col = st.selectbox("Select Image URL Column", df.columns)
        name_col = st.selectbox("Select Filename/ID Column", df.columns)
        
        if st.button("Process Batch"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create a ZIP file in memory
            import zipfile
            zip_buffer = BytesIO()
            
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for i, row in df.iterrows():
                    try:
                        url = row[url_col]
                        fname = str(row[name_col])
                        
                        status_text.text(f"Processing {i+1}/{len(df)}: {fname}...")
                        
                        # Download
                        resp = requests.get(url, timeout=10)
                        img = load_and_orient_image(BytesIO(resp.content))
                        
                        # Process
                        final = process_final_image(img, target_size, padding, output_format, quality)
                        
                        # Save to ZIP
                        img_byte_arr = BytesIO()
                        save_fmt = "JPEG" if output_format == "JPEG" else "PNG"
                        ext = save_fmt.lower()
                        final.save(img_byte_arr, format=save_fmt, quality=quality)
                        
                        zf.writestr(f"{fname}.{ext}", img_byte_arr.getvalue())
                        
                    except Exception as e:
                        st.warning(f"Skipped {fname}: {e}")
                    
                    progress_bar.progress((i + 1) / len(df))
            
            status_text.success("Batch Complete!")
            st.download_button(
                "Download All as ZIP", 
                zip_buffer.getvalue(), 
                file_name="processed_images.zip", 
                mime="application/zip"
            )

# 3. BATCH (LOCAL FILES)
elif processing_mode == "Batch (Local Files)":
    st.subheader("Batch Processing (Local Upload)")
    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png", "webp"])
    
    if uploaded_files and st.button("Process Batch"):
        progress_bar = st.progress(0)
        
        import zipfile
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for i, file in enumerate(uploaded_files):
                try:
                    img = load_and_orient_image(file)
                    final = process_final_image(img, target_size, padding, output_format, quality)
                    
                    # Save to ZIP
                    img_byte_arr = BytesIO()
                    save_fmt = "JPEG" if output_format == "JPEG" else "PNG"
                    ext = save_fmt.lower()
                    final.save(img_byte_arr, format=save_fmt, quality=quality)
                    
                    # Use original filename but with new extension
                    original_name = file.name.rsplit('.', 1)[0]
                    zf.writestr(f"{original_name}.{ext}", img_byte_arr.getvalue())
                    
                except Exception as e:
                    st.error(f"Error on {file.name}: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                
        st.success("Done!")
        st.download_button(
            "Download All as ZIP", 
            zip_buffer.getvalue(), 
            file_name="processed_batch.zip", 
            mime="application/zip"
        )