from ultralytics import YOLO
import os
import shutil
import glob

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
    os.makedirs('models', exist_ok=True)
    
    if os.path.exists('models/best.pt'):
        print("✅ Model already exists!")
        return
    
    print("🚀 Training agricultural crop-weed detection model...")
    
    split_dataset()
    
    data_yaml_content = """
path: ../data/agri_data
train: train/images
val: val/images
test: test/images
nc: 2
names: ['crop', 'weed']
"""
    os.makedirs('data', exist_ok=True)
    with open('data/data.yaml', 'w') as f:
        f.write(data_yaml_content)
    
    model = YOLO('yolov5s.pt')
    model.train(
        data='data/data.yaml',
        epochs=10,          # 10 epochs as requested
        imgsz=640,
        batch=16,
        project='outputs',
        name='agri_detection',
        exist_ok=True
    )
    
    weights_path = 'outputs/agri_detection/weights/best.pt'
    if os.path.exists(weights_path):
        shutil.copy(weights_path, 'models/best.pt')
        print("✅ Training completed! Model saved.")
    else:
        print("❌ Training failed!")

if __name__ == "__main__":
    train_model()
