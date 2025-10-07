import streamlit as st
from ultralytics import YOLO
import subprocess
import os
from PIL import Image
import numpy as np

# Paths
SCRIPTS_TRAIN = 'scripts/train.py'
SCRIPTS_INFER = 'scripts/infer.py'
ROOT_TRAIN = 'train.py'
ROOT_INFER = 'infer.py'
MODEL_PATH = 'models/best.pt'

def run_external_script(script_path, extra_args=None):
    """Run an external python script and show logs in Streamlit."""
    cmd = ['python', script_path]
    if extra_args:
        cmd += extra_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        st.error(f"Error running {script_path}:\n{result.stderr}")
    else:
        st.success(f"Successfully ran {script_path}")
        if result.stdout:
            st.text(result.stdout)

@st.cache_resource(show_spinner=True)
def train_and_setup_model():
    # Run script-level train.py first
    if os.path.exists(SCRIPTS_TRAIN):
        st.info("Running scripts/train.py to train and setup model...")
        run_external_script(SCRIPTS_TRAIN)
    else:
        st.warning("scripts/train.py not found, skipping.")

    # Setup model for inference
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model weights not found at {MODEL_PATH}. Training may have failed.")
        return None
    else:
        st.success("Loaded trained model weights.")
        return YOLO(MODEL_PATH)

def classify_and_display(model, image):
    results = model(image)
    result_obj = results[0]  
    detected_table = result_obj.to_df().to_pandas()  # Convert Polars DF to Pandas DF
    img_with_boxes = np.array(result_obj.plot())
    st.image(img_with_boxes, caption="Detection Result", use_column_width=True)

    crop_count, weed_count = 0, 0
    crop_conf, weed_conf = 0, 0
    for _, row in detected_table.iterrows():
        class_name = row.get('name', 'unknown')
        conf = row.get('confidence', row.get('conf', 0))
        if class_name == 'crop':
            crop_count += 1
            crop_conf += conf
        elif class_name == 'weed':
            weed_count += 1
            weed_conf += conf

    # Message for user
    if crop_count == 0 and weed_count == 0:
        st.warning("❓ No crops or weeds detected in this image.")
    elif crop_count > weed_count:
        avg = crop_conf / crop_count if crop_count else 0
        st.success(f"🌾 This is a **CROP** image\n({crop_count} crops detected, avg confidence: {avg:.1%})")
    elif weed_count > crop_count:
        avg = weed_conf / weed_count if weed_count else 0
        st.error(f"🌿 This is a **WEED** image\n({weed_count} weeds detected, avg confidence: {avg:.1%})")
    else:
        avg_crop = crop_conf/crop_count if crop_count else 0
        avg_weed = weed_conf/weed_count if weed_count else 0
        if avg_crop >= avg_weed:
            st.success(f"🌾 This is a **CROP** image (higher avg confidence: {avg_crop:.1%})")
        else:
            st.error(f"🌿 This is a **WEED** image (higher avg confidence: {avg_weed:.1%})")
    st.write(detected_table)


def main():
    st.title("🌱 Crop and Weed Detector • End-to-End Automation")
    st.markdown("*Training, inference, and interactive detection all from this UI.*")

    # Train and setup model (runs train.py automatically!)
    model = train_and_setup_model()
    if model is None:
        st.stop()

    # Run scripts/infer.py and root infer.py automatically (could be batch or demo runs)
    if os.path.exists(SCRIPTS_INFER):
        st.info("Running scripts/infer.py for batch/test inference...")
        run_external_script(SCRIPTS_INFER)
    else:
        st.warning("scripts/infer.py not found, skipping.")

    # Interactive image upload
    uploaded_file = st.file_uploader("Upload your agricultural image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Running model detection..."):
            classify_and_display(model, image)

if __name__ == "__main__":
    main()