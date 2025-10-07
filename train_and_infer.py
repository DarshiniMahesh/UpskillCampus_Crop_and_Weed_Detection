from ultralytics import YOLO
import os
import sys
import shutil
import glob
import matplotlib.pyplot as plt
from PIL import Image

def split_dataset():
    """Split dataset into train/val/test"""
    data_dir = 'data/agri_data/data'
    
    for split in ['train', 'val', 'test']:
        os.makedirs(f'data/agri_data/{split}/images', exist_ok=True)
        os.makedirs(f'data/agri_data/{split}/labels', exist_ok=True)
    
    image_files = glob.glob(f'{data_dir}/*.jpeg') + glob.glob(f'{data_dir}/*.jpg') + glob.glob(f'{data_dir}/*.png')
    total = len(image_files)
    train_split = int(0.8 * total)
    val_split = int(0.9 * total)
    
    print(f"Splitting {total} images: Train={train_split}, Val={val_split-train_split}, Test={total-val_split}")
    
    for i, img_path in enumerate(image_files):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(data_dir, f'{base_name}.txt')
        
        split = 'train' if i < train_split else 'val' if i < val_split else 'test'
        
        shutil.copy(img_path, f'data/agri_data/{split}/images/')
        if os.path.exists(txt_path):
            shutil.copy(txt_path, f'data/agri_data/{split}/labels/')

def train_model():
    """Train the agricultural detection model"""
    os.makedirs('models', exist_ok=True)
    
    if os.path.exists('models/best.pt'):
        print("✅ Model already exists!")
        return
    
    print("🚀 Training agricultural crop-weed detection model...")
    
    # Prepare dataset
    split_dataset()
    
    # Create data.yaml
    data_yaml_content = """
path: ../data/agri_data
train: train/images
val: val/images
test: test/images

nc: 2
names: ['crop', 'weed']
"""
    with open('data/data.yaml', 'w') as f:
        f.write(data_yaml_content)
    
    # Train model
    model = YOLO('yolov5s.pt')
    model.train(
        data='data/data.yaml',
        epochs=10,
        imgsz=640,
        batch=16,
        project='outputs',
        name='agri_detection',
        exist_ok=True
    )
    
    # Save weights
    weights_path = 'outputs/agri_detection/weights/best.pt'
    if os.path.exists(weights_path):
        shutil.copy(weights_path, 'models/best.pt')
        print("✅ Training completed! Model saved.")
    else:
        print("❌ Training failed!")

def classify_and_infer(image_path):
    """Run inference and classify the image"""
    if not os.path.exists('models/best.pt'):
        print("❌ Model not found! Train first.")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"🔍 Analyzing {image_path}...")
    
    # Load model and predict
    model = YOLO('models/best.pt')
    results = model.predict(image_path, conf=0.25)
    
    # Count detections
    crop_count = 0
    weed_count = 0
    crop_conf = 0
    weed_conf = 0
    
    for r in results:
        # Display image with detections
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        
        plt.figure(figsize=(12, 8))
        plt.imshow(im)
        plt.axis('off')
        plt.tight_layout()
        
        # Count objects
        if len(r.boxes) > 0:
            for box in r.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                
                if cls == 0:  # crop
                    crop_count += 1
                    crop_conf += conf
                else:  # weed
                    weed_count += 1
                    weed_conf += conf
    
    # Determine classification
    print("\n" + "="*60)
    if crop_count == 0 and weed_count == 0:
        print("❓ RESULT: No crops or weeds detected")
        plt.title("No Detection", fontsize=16, fontweight='bold')
    elif crop_count > weed_count:
        avg_conf = crop_conf / crop_count if crop_count > 0 else 0
        print(f"🌾 RESULT: This is a CROP!")
        print(f"   📊 {crop_count} crops detected with {avg_conf:.1%} average confidence")
        plt.title("CLASSIFICATION: CROP 🌾", fontsize=16, fontweight='bold', color='green')
    elif weed_count > crop_count:
        avg_conf = weed_conf / weed_count if weed_count > 0 else 0
        print(f"🌿 RESULT: This is a WEED!")
        print(f"   📊 {weed_count} weeds detected with {avg_conf:.1%} average confidence")
        plt.title("CLASSIFICATION: WEED 🌿", fontsize=16, fontweight='bold', color='red')
    else:
        avg_crop = crop_conf / crop_count if crop_count > 0 else 0
        avg_weed = weed_conf / weed_count if weed_count > 0 else 0
        if avg_crop > avg_weed:
            print(f"🌾 RESULT: This is a CROP (higher confidence: {avg_crop:.1%})")
            plt.title("CLASSIFICATION: CROP 🌾", fontsize=16, fontweight='bold', color='green')
        else:
            print(f"🌿 RESULT: This is a WEED (higher confidence: {avg_weed:.1%})")
            plt.title("CLASSIFICATION: WEED 🌿", fontsize=16, fontweight='bold', color='red')
    
    print("="*60)
    plt.show()

if __name__ == "__main__":
    # Train model
    train_model()
    
    # Run inference if image provided
    if len(sys.argv) > 1:
        classify_and_infer(sys.argv[1])
    else:
        print("💡 Usage: python train_and_infer.py path/to/image.jpg")
        print("   Or just run: python train_and_infer.py (to train only)")
