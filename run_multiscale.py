import os
import random
import cv2
import numpy as np
import yaml
import shutil
from ultralytics import YOLO

# --- Configuration Commune ---
BACKGROUNDS_DIR = 'backgrounds_urbains'
CUTOUTS_DIR = 'dataset_rogner'
NUM_IMAGES_PER_LEVEL = 1000 # 1000 images par niveau pour aller vite
IMAGE_SIZE = (640, 640)
MIN_OBJECTS_PER_IMG = 1
MAX_OBJECTS_PER_IMG = 6
EPOCHS_PER_LEVEL = 10

FILENAME_TO_CLASS = {
    'condor.png': 'condor',
    'hangar.png': 'hangar',
    'helico.png': 'helicopter',
    'jammer.png': 'jammer',
    'jet_plane.png': 'jet_plane',
    'large launcher.png': 'large_launcher',
    'large_tower.png': 'large_tower',
    'medium_launcher.png': 'medium_launcher',
    'medium_plane.png': 'medium_plane',
    'mine_roller.png': 'mine_roller',
    'small_launcher.png': 'small_launcher',
    'small_plane.png': 'small_plane',
    'small_tower.png': 'small_tower',
    'space_craft.png': 'spacecraft',
    'tank.png': 'tank',
    'tata.png': 'ta-ta'
}
CLASSES = list(FILENAME_TO_CLASS.values())
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

def generate_dataset(output_dir, min_scale, max_scale, num_images):
    print(f"\n GENERATIN DATASET{output_dir} (SCALE: {min_scale} a {max_scale}) ---")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    os.makedirs(f'{output_dir}/images/train', exist_ok=True)
    os.makedirs(f'{output_dir}/labels/train', exist_ok=True)
    os.makedirs(f'{output_dir}/images/val', exist_ok=True)
    os.makedirs(f'{output_dir}/labels/val', exist_ok=True)

    cutouts = []
    for f in os.listdir(CUTOUTS_DIR):
        if not f.endswith('.png'): continue
        if f not in FILENAME_TO_CLASS: continue
        path = os.path.join(CUTOUTS_DIR, f)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[2] != 4: continue
        cutouts.append({'img': img, 'class_id': CLASS_TO_ID[FILENAME_TO_CLASS[f]], 'name': f})

    backgrounds = [os.path.join(BACKGROUNDS_DIR, f) for f in os.listdir(BACKGROUNDS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def random_transform(img_bgra):
        scale = random.uniform(min_scale, max_scale)
        angle = random.uniform(0, 360)
        h, w = img_bgra.shape[:2]
        diag = int(np.sqrt(h*h + w*w))
        new_img = np.zeros((diag, diag, 4), dtype=np.uint8)
        start_y = (diag - h) // 2
        start_x = (diag - w) // 2
        new_img[start_y:start_y+h, start_x:start_x+w] = img_bgra
        M = cv2.getRotationMatrix2D((diag/2, diag/2), angle, scale)
        rotated = cv2.warpAffine(new_img, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        alpha = rotated[:, :, 3]
        coords = cv2.findNonZero(alpha)
        if coords is None: return img_bgra
        x, y, w_obj, h_obj = cv2.boundingRect(coords)
        cropped = rotated[y:y+h_obj, x:x+w_obj]
        if random.random() > 0.5: cropped = cv2.flip(cropped, 1)
        if random.random() > 0.5: cropped = cv2.flip(cropped, 0)
        return cropped

    def overlay_transparent(bg_img, overlay_t, x, y):
        h, w = overlay_t.shape[:2]
        bg_h, bg_w = bg_img.shape[:2]
        if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0: return False
        start_y, start_x = max(0, y), max(0, x)
        end_y, end_x = min(bg_h, y + h), min(bg_w, x + w)
        overlay_start_y = max(0, -y)
        overlay_start_x = max(0, -x)
        overlay_end_y = overlay_start_y + (end_y - start_y)
        overlay_end_x = overlay_start_x + (end_x - start_x)
        overlay_crop = overlay_t[overlay_start_y:overlay_end_y, overlay_start_x:overlay_end_x]
        alpha = overlay_crop[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        for c in range(3):
            bg_img[start_y:end_y, start_x:end_x, c] = (alpha * overlay_crop[:, :, c] + alpha_inv * bg_img[start_y:end_y, start_x:end_x, c])
        return True

    for i in range(num_images):
        bg_path = random.choice(backgrounds)
        bg_full = cv2.imread(bg_path)
        if bg_full is None: continue
        bh, bw = bg_full.shape[:2]
        if bh > IMAGE_SIZE[1] and bw > IMAGE_SIZE[0]:
            y_offset = random.randint(0, bh - IMAGE_SIZE[1])
            x_offset = random.randint(0, bw - IMAGE_SIZE[0])
            bg = bg_full[y_offset:y_offset+IMAGE_SIZE[1], x_offset:x_offset+IMAGE_SIZE[0]].copy()
        else:
            bg = cv2.resize(bg_full, IMAGE_SIZE).copy()
            
        if random.random() > 0.5:
            bg = cv2.convertScaleAbs(bg, alpha=random.uniform(0.8, 1.2), beta=random.randint(-20, 20))
        
        num_objs = random.randint(MIN_OBJECTS_PER_IMG, MAX_OBJECTS_PER_IMG)
        labels = []
        for _ in range(num_objs):
            obj = random.choice(cutouts)
            trans_obj = random_transform(obj['img'])
            h_obj, w_obj = trans_obj.shape[:2]
            
            if w_obj >= IMAGE_SIZE[0] or h_obj >= IMAGE_SIZE[1]:
                continue
                
            x = random.randint(0, IMAGE_SIZE[0] - w_obj)
            y = random.randint(0, IMAGE_SIZE[1] - h_obj)
            if overlay_transparent(bg, trans_obj, x, y):
                cx = (x + w_obj / 2) / IMAGE_SIZE[0]
                cy = (y + h_obj / 2) / IMAGE_SIZE[1]
                nw = w_obj / IMAGE_SIZE[0]
                nh = h_obj / IMAGE_SIZE[1]
                labels.append(f"{obj['class_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                
        split = 'train' if random.random() > 0.15 else 'val'
        cv2.imwrite(f"{output_dir}/images/{split}/{i:05d}.jpg", bg)
        with open(f"{output_dir}/labels/{split}/{i:05d}.txt", "w") as f:
            f.write("\n".join(labels))

    yaml_content = {'path': os.path.abspath(output_dir), 'train': 'images/train', 'val': 'images/val', 'nc': len(CLASSES), 'names': CLASSES}
    with open(f"{output_dir}/data.yaml", "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"Dataset {output_dir} pret !")


def main():
    # 1. Dataset L2 (Max Zoom )
    generate_dataset("dataset_L2", min_scale=0.8, max_scale=2.0, num_images=NUM_IMAGES_PER_LEVEL)
    
    # 2. Dataset L1 (Medium Zoom )
    generate_dataset("dataset_L1", min_scale=0.3, max_scale=0.8, num_images=NUM_IMAGES_PER_LEVEL)
    
    # 3. Dataset L0 (Vue Globale )
    generate_dataset("dataset_L0", min_scale=0.05, max_scale=0.3, num_images=NUM_IMAGES_PER_LEVEL)

    # CURICULUM LEARNING
    print("\n Training L1 (From an initial model)")
    if os.path.exists("yolo_L1_best.pt"):
        initial_weights = "yolo_L1_best.pt"
    elif os.path.exists("yolov8n-visdrone.pt"):
        initial_weights = "yolov8n-visdrone.pt"
    else:
        print("Warning: Aerial pre-trained weights not found. Falling back to generic COCO (yolov8n.pt).")
        initial_weights = "yolov8n.pt"
        
    print(f"Initializing model with weights: {initial_weights}")

    model_L2 = YOLO(initial_weights)
    results_L2 = model_L2.train(data="dataset_L2/data.yaml", epochs=EPOCHS_PER_LEVEL, imgsz=640, batch=2, workers=0, device=0, name="yolo_L2_run")
    best_L2_path = results_L2.top1_weights if hasattr(results_L2, 'top1_weights') else str(model_L2.trainer.best)
    shutil.copy(best_L2_path, "model_L2.pt")
    print(f"L2 model registereD under model_L2.pt")

    print("\n Training L1 ( From L2)")
    model_L1 = YOLO("model_L2.pt")
    results_L1 = model_L1.train(data="dataset_L1/data.yaml", epochs=EPOCHS_PER_LEVEL, imgsz=640, batch=2, workers=0, device=0, name="yolo_L1_run")
    best_L1_path = results_L1.top1_weights if hasattr(results_L1, 'top1_weights') else str(model_L1.trainer.best)
    shutil.copy(best_L1_path, "model_L1.pt")
    print(f"L1 model registered under model_L1.pt")

    print("\n Training L0 ( from L1 model) ---")
    model_L0 = YOLO("model_L1.pt")
    results_L0 = model_L0.train(data="dataset_L0/data.yaml", epochs=EPOCHS_PER_LEVEL, imgsz=640, batch=2, workers=0, device=0, name="yolo_L0_run")
    best_L0_path = results_L0.top1_weights if hasattr(results_L0, 'top1_weights') else str(model_L0.trainer.best)
    shutil.copy(best_L0_path, "model_L0.pt")
    print(f"L0 Model registered under model_L0.pt")
    
    print("\n=== PIPELINE ACCOMPLISHED ===")
    print(" final models : model_L2.pt, model_L1.pt, model_L0.pt")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
