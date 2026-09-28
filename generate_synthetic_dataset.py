import os
import random
import cv2
import numpy as np
import yaml
from pathlib import Path

# --- Configuration ---
BACKGROUNDS_DIR = 'backgrounds_urbains'
CUTOUTS_DIR = 'dataset_rogner'
OUTPUT_DIR = 'dataset_synthetic'
NUM_IMAGES = 2000 # On génère 2000 images synthétiques
IMAGE_SIZE = (640, 640)
MIN_OBJECTS_PER_IMG = 1
MAX_OBJECTS_PER_IMG = 6

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

CLASSES = [
    'condor', 'hangar', 'helicopter', 'jammer', 'jet_plane', 'large_launcher', 
    'large_tower', 'medium_launcher', 'medium_plane', 'mine_roller', 
    'small_launcher', 'small_plane', 'small_tower', 'spacecraft', 'ta-ta', 'tank'
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

def load_cutouts():
    cutouts = []
    for f in os.listdir(CUTOUTS_DIR):
        if not f.endswith('.png'): continue
        if f not in FILENAME_TO_CLASS:
            print(f"Warning: {f} non mappé.")
            continue
        path = os.path.join(CUTOUTS_DIR, f)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[2] != 4:
            print(f"Warning: {f} n'a pas de canal alpha.")
            continue
        cls_name = FILENAME_TO_CLASS[f]
        cutouts.append({'img': img, 'class_id': CLASS_TO_ID[cls_name], 'name': f})
    return cutouts

def load_backgrounds():
    if not os.path.exists(BACKGROUNDS_DIR):
        os.makedirs(BACKGROUNDS_DIR)
        return []
    
    bg_files = [os.path.join(BACKGROUNDS_DIR, f) for f in os.listdir(BACKGROUNDS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return bg_files

def random_transform(img_bgra):
    # Scale
    scale = random.uniform(0.4, 1.2)
    # Rotation
    angle = random.uniform(0, 360)
    
    h, w = img_bgra.shape[:2]
    
    # Calculate new bounds
    diag = int(np.sqrt(h*h + w*w))
    new_img = np.zeros((diag, diag, 4), dtype=np.uint8)
    
    # Place in center
    start_y = (diag - h) // 2
    start_x = (diag - w) // 2
    new_img[start_y:start_y+h, start_x:start_x+w] = img_bgra
    
    # Rotate
    M = cv2.getRotationMatrix2D((diag/2, diag/2), angle, scale)
    rotated = cv2.warpAffine(new_img, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    
    # Crop to non-transparent pixels
    alpha = rotated[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return img_bgra # Fallback
        
    x, y, w_obj, h_obj = cv2.boundingRect(coords)
    cropped = rotated[y:y+h_obj, x:x+w_obj]
    
    # Optional: Random horizontal/vertical flip
    if random.random() > 0.5:
        cropped = cv2.flip(cropped, 1)
    if random.random() > 0.5:
        cropped = cv2.flip(cropped, 0)
        
    return cropped

def overlay_transparent(bg_img, overlay_t, x, y):
    h, w = overlay_t.shape[:2]
    bg_h, bg_w = bg_img.shape[:2]
    
    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return False
        
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
        bg_img[start_y:end_y, start_x:end_x, c] = (
            alpha * overlay_crop[:, :, c] + alpha_inv * bg_img[start_y:end_y, start_x:end_x, c]
        )
    return True

def main():
    print("Chargement des détourages...")
    cutouts = load_cutouts()
    if not cutouts:
        print("Erreur: Aucun détourage trouvé.")
        return
        
    print("Chargement des fonds...")
    backgrounds = load_backgrounds()
    if not backgrounds:
        print(f"ATTENTION: Le dossier '{BACKGROUNDS_DIR}' est vide. Ajoutez des images (villes, toits, routes) et relancez.")
        return
        
    # Setup directories
    os.makedirs(f'{OUTPUT_DIR}/images/train', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/labels/train', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/images/val', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/labels/val', exist_ok=True)
    
    print(f"Génération de {NUM_IMAGES} images...")
    for i in range(NUM_IMAGES):
        bg_path = random.choice(backgrounds)
        bg_full = cv2.imread(bg_path)
        if bg_full is None: continue
        
        # Random crop of background to IMAGE_SIZE
        bh, bw = bg_full.shape[:2]
        if bh > IMAGE_SIZE[1] and bw > IMAGE_SIZE[0]:
            y_offset = random.randint(0, bh - IMAGE_SIZE[1])
            x_offset = random.randint(0, bw - IMAGE_SIZE[0])
            bg = bg_full[y_offset:y_offset+IMAGE_SIZE[1], x_offset:x_offset+IMAGE_SIZE[0]].copy()
        else:
            bg = cv2.resize(bg_full, IMAGE_SIZE).copy()
            
        # Optional: blur/color change background slightly to augment
        if random.random() > 0.5:
            bg = cv2.convertScaleAbs(bg, alpha=random.uniform(0.8, 1.2), beta=random.randint(-20, 20))
        
        num_objs = random.randint(MIN_OBJECTS_PER_IMG, MAX_OBJECTS_PER_IMG)
        labels = []
        
        for _ in range(num_objs):
            obj = random.choice(cutouts)
            trans_obj = random_transform(obj['img'])
            h_obj, w_obj = trans_obj.shape[:2]
            
            # Random position
            x = random.randint(0, IMAGE_SIZE[0] - w_obj)
            y = random.randint(0, IMAGE_SIZE[1] - h_obj)
            
            if overlay_transparent(bg, trans_obj, x, y):
                # YOLO format: class x_center y_center width height (normalized)
                cx = (x + w_obj / 2) / IMAGE_SIZE[0]
                cy = (y + h_obj / 2) / IMAGE_SIZE[1]
                nw = w_obj / IMAGE_SIZE[0]
                nh = h_obj / IMAGE_SIZE[1]
                labels.append(f"{obj['class_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                
        # Split train/val
        split = 'train' if random.random() > 0.15 else 'val'
        
        img_filename = f"{i:05d}.jpg"
        lbl_filename = f"{i:05d}.txt"
        
        cv2.imwrite(f"{OUTPUT_DIR}/images/{split}/{img_filename}", bg)
        with open(f"{OUTPUT_DIR}/labels/{split}/{lbl_filename}", "w") as f:
            f.write("\n".join(labels))
            
        if (i+1) % 100 == 0:
            print(f"{i+1}/{NUM_IMAGES} générées...")

    # Write data.yaml
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASSES),
        'names': CLASSES
    }
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        
    print(f"Génération terminée. Dataset enregistré dans {OUTPUT_DIR}.")

if __name__ == "__main__":
    main()
