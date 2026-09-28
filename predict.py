import base64
import math
import threading
import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
from utils import view_bbox_to_global, clip_bbox_to_frame
from dtos import OBJECT_CLASSES, DroneFlybyPredictResponseDto, DroneFlybyPredictionDto, RequestedViewDto, FULL_FRAME_CENTER

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOADING 3 SPECIALIZED MODELS
print("Chargement des modeles multi-echelles en memoire (cela peut prendre quelques secondes)...")
try:
    model_L0 = YOLO("model_L0.pt")
    model_L1 = YOLO("model_L1.pt")
    model_L2 = YOLO("model_L2.pt")
except Exception as e:
    print(f"ATTENTION : Un ou plusieurs modeles manquants. Erreur: {e}")
    # Fallbacks 
    model_L0 = YOLO("yolo_L1_best.pt")
    model_L1 = YOLO("yolo_L1_best.pt")
    model_L2 = YOLO("yolo_L1_best.pt")

TRAINED_CLASSES = [model_L0.names[i] for i in range(len(model_L0.names))]

# WARM-UP
dummy_img = np.zeros((540, 960, 3), dtype=np.uint8)
model_L0.predict(dummy_img, verbose=False, imgsz=640)
model_L1.predict(dummy_img, verbose=False, imgsz=640)
model_L2.predict(dummy_img, verbose=False, imgsz=640)
print("Les 3 modeles sont prechauffes !")

ACTIVE_TRACKS = []
NEXT_TRACK_ID = 0
LAST_SEQUENCE_ID = None

# SWEEPING L0 -> L1 -> L0 -> L1 for trying but I hadn't the tiem to go futher
DWELL_FRAMES = 2
SWEEP_POSITIONS = [
    (0, 1920, 1080),  # L0 milieu
    (1, 2880, 540),   # L1 haut-droite
    (0, 1920, 1080),  # L0 milieu
    (1, 960, 540),    # L1 haut-gauche
]
FRAME_IN_CYCLE = 0

STATE_LOCK = threading.Lock()
YOLO_CONF = 0.15
MIN_REPORT_CONF = 0.1

def box_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / float(box1_area + box2_area - intersection_area)

def predict(request):
    global ACTIVE_TRACKS, NEXT_TRACK_ID, LAST_SEQUENCE_ID, FRAME_IN_CYCLE

    cur_level = request.view.resolution_level

    with STATE_LOCK:
        try:
            if request.sequence_id != LAST_SEQUENCE_ID:
                ACTIVE_TRACKS = []
                LAST_SEQUENCE_ID = request.sequence_id
                FRAME_IN_CYCLE = 0

            # DECODE IMAGE 
            img_bytes = base64.b64decode(request.frame)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Model Selection depending on the Zoom
            if cur_level == 0:
                active_model = model_L0
            elif cur_level == 1:
                active_model = model_L1
            else:
                active_model = model_L2

            # PREDICTION
            results = active_model.predict(img, conf=YOLO_CONF, verbose=False, imgsz=640)

            current_detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id < len(TRAINED_CLASSES):
                        obj_name = TRAINED_CLASSES[cls_id]
                        if obj_name not in OBJECT_CLASSES: continue
                        view_bbox = box.xyxy[0].tolist()
                        view_bbox_norm = [
                            view_bbox[0] / request.view.width,
                            view_bbox[1] / request.view.height,
                            view_bbox[2] / request.view.width,
                            view_bbox[3] / request.view.height
                        ]
                        global_box = view_bbox_to_global(
                            view_bbox_norm,
                            request.view.source_region_xyxy,
                            request.original_width,
                            request.original_height
                        )
                        clipped_box = clip_bbox_to_frame(global_box)
                        if clipped_box:
                            current_detections.append({
                                "object_id": obj_name,
                                "bbox": clipped_box,
                                "confidence": conf
                            })

            # TRACKING
            matched_track_ids = set()
            for det in current_detections:
                best_dist = float('inf')
                best_track = None
                for t in ACTIVE_TRACKS:
                    if t["id"] not in matched_track_ids and t["object_id"] == det["object_id"]:
                        pred_cx = (t["bbox"][0] + t["bbox"][2]) / 2
                        pred_cy = (t["bbox"][1] + t["bbox"][3]) / 2
                        new_cx = (det["bbox"][0] + det["bbox"][2]) / 2
                        new_cy = (det["bbox"][1] + det["bbox"][3]) / 2
                        dist_pixels = math.hypot((new_cx - pred_cx) * request.original_width,
                                                 (new_cy - pred_cy) * request.original_height)
                        if dist_pixels < best_dist:
                            best_dist = dist_pixels
                            best_track = t
                tolerance = 150.0
                if best_track is not None and best_dist < tolerance:
                    matched_track_ids.add(best_track["id"])
                    best_track["bbox"] = det["bbox"]
                    best_track["confidence"] = max(best_track["confidence"], det["confidence"])
                    best_track["age"] = 0
                else:
                    ACTIVE_TRACKS.append({"id": NEXT_TRACK_ID, "object_id": det["object_id"], "bbox": det["bbox"], "confidence": det["confidence"], "age": 0})
                    NEXT_TRACK_ID += 1

            # Global NMS 
            ACTIVE_TRACKS.sort(key=lambda x: x["confidence"], reverse=True)
            nms_tracks = []
            for t in ACTIVE_TRACKS:
                keep = True
                for kept in nms_tracks:
                    if box_iou(t["bbox"], kept["bbox"]) > 0.40:
                        keep = False
                        break
                if keep: nms_tracks.append(t)
            ACTIVE_TRACKS = nms_tracks

            annotations = [DroneFlybyPredictionDto(object_id=t["object_id"], bbox=t["bbox"], confidence=t["confidence"]) for t in ACTIVE_TRACKS if t["confidence"] > MIN_REPORT_CONF]

            # CAM PILOT
            n_states = len(SWEEP_POSITIONS)
            state_idx = (FRAME_IN_CYCLE // DWELL_FRAMES) % n_states
            position_in_state = FRAME_IN_CYCLE % DWELL_FRAMES
            FRAME_IN_CYCLE += 1

            requested_view_obj = None
            if position_in_state == 0:
                level, cx, cy = SWEEP_POSITIONS[state_idx]
                req_level, req_cx, req_cy = make_safe_camera_command(request, level, cx, cy)
                requested_view_obj = RequestedViewDto(resolution_level=req_level, center_x=req_cx, center_y=req_cy)

            return DroneFlybyPredictResponseDto(request_id=request.request_id, frame=request.frame, annotations=annotations, requested_view=requested_view_obj)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return DroneFlybyPredictResponseDto(request_id=request.request_id, frame=request.frame, annotations=[], requested_view=RequestedViewDto(resolution_level=0, center_x=FULL_FRAME_CENTER[0], center_y=FULL_FRAME_CENTER[1]))

def make_safe_camera_command(request, req_level, req_cx, req_cy):
    real_cx = request.view.center_x
    real_cy = request.view.center_y
    constraints = request.camera_constraints
    if req_level not in constraints.allowed_resolution_levels: return 0, FULL_FRAME_CENTER[0], FULL_FRAME_CENTER[1]
    if req_level != 0:
        bounds = constraints.bounds_for_level(req_level)
        if bounds:
            req_cx = min(max(req_cx, bounds.minimum_center_x), bounds.maximum_center_x)
            req_cy = min(max(req_cy, bounds.minimum_center_y), bounds.maximum_center_y)
    is_full_reset = (req_level == 0 and req_cx == FULL_FRAME_CENTER[0] and req_cy == FULL_FRAME_CENTER[1])
    if not (is_full_reset and constraints.full_view_reset_exempt_from_delta):
        max_delta = constraints.maximum_center_delta
        dist = math.hypot(req_cx - real_cx, req_cy - real_cy)
        if dist > max_delta: return 0, FULL_FRAME_CENTER[0], FULL_FRAME_CENTER[1]
    return int(req_level), int(req_cx), int(req_cy)
