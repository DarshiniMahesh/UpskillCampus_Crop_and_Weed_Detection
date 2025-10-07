from ultralytics import YOLO
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt

def classify_and_infer(image_path):
    model_path = 'models/best.pt'
    
    if not os.path.exists(model_path):
        print("❌ Model not found! Please train the model first using train.py.")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"🔍 Running inference on {image_path}...")
    
    model = YOLO(model_path)
    results = model.predict(image_path, conf=0.25)
    
    crop_count = 0
    weed_count = 0
    crop_conf = 0
    weed_conf = 0
    
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        
        plt.figure(figsize=(12, 8))
        plt.imshow(im)
        plt.axis('off')
        plt.tight_layout()
        
        if len(r.boxes) > 0:
            for box in r.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                
                if cls == 0:
                    crop_count += 1
                    crop_conf += conf
                else:
                    weed_count += 1
                    weed_conf += conf
    
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
    if len(sys.argv) > 1:
        classify_and_infer(sys.argv[1])
    else:
        print("Usage: python scripts/infer.py path/to/image.jpg")
