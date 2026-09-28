# Drone Flyby: Autonomous Multi-Scale Object Detection & Camera Policy

## Project Overview
This repository contains my solution for the **Nordic AI Cup 2026 Challenge**. The objective was to build an autonomous pipeline for a survey drone flying at 600m altitude. The system had to detect 16 classes of small objects (vehicles, aircraft, towers) in real-time (333ms per frame) while dynamically steering a limited-bandwidth  camera across a 4K landscape.

> **Challenge Rules:** The full instructions and API constraints of the hackathon can be found in `CHALLENGE_INSTRUCTIONS.md`.

## Architecture & Engineering Choices

Due to the strict real-time constraints and the lack of a large-scale annotated dataset for this specific aerial viewpoint, I architected a highly optimized pipeline focusing on low latency and progressive learning:

### 1. Data Engineering & Curriculum Learning
Since small objects are heavily distorted across different zoom levels, I built a custom synthetic dataset generator and a multi-scale training pipeline (`run_multiscale.py`).
*   **Altitude-Matched Backgrounds:** I sourced diverse aerial landscapes from Google Earth, **strictly captured at a 600m altitude** to maintain the correct Ground Sample Distance (GSD) physics.
*   **Curriculum Learning (Cascade Training):** Instead of training a single model from scratch, I applied a Curriculum Learning strategy. I generated 3 synthetic datasets (L2: Large objects, L1: Medium, L0: Tiny). I first trained YOLOv8 on L2, then transferred those learned weights to train L1, and finally used the L1 weights to train L0. This allowed the network to learn rich structural features before tackling severe downsampling.

### 2. Real-Time Latency Management & Asynchronous Tracking
With a strict 333ms frame emission rate, heavy YOLO inferences inevitably cause skipped frames. I designed the tracking system (`predict.py`) specifically to absorb this latency:
*   **Time-Aware Kinematic Tracking:** The tracker calculates `frames_elapsed` between requests. If the YOLO inference takes too long and the server drops frames, the kinematic equations automatically fast-forward the bounding boxes to their correct current positions using the drone's velocity.
*   **Optical Flow Calibration:** I utilized Farneback Optical Flow during L0 (Full-frame) sweeps to dynamically estimate the drone's vertical velocity (`GLOBAL_VY`).
*   **Global Temporal NMS:** Applied Non-Maximum Suppression across active tracks to prevent duplicate reporting of the same object over time.

### 3. The Camera Steering Policy: "Top-Band Sweep" Heuristic
Balancing camera movement and detection accuracy is the core of this challenge. 
*   **Hackathon Implementation:** Because the drone flies forward in a straight line, *new* terrain only enters from the top of the frame. I implemented a deterministic sweep oscillating between L0 (Full Frame) and L1 (Top-Left / Top-Right). The bottom of the frame is never scanned by the camera; instead, objects detected at the top are tracked kinematically as they move down the screen, drastically saving inference compute.
*   **Future Work (Dynamic ROI Zooming):** My ultimate architectural goal for this project is to replace the deterministic sweep with a dynamic heuristic policy: keeping the camera at L0 to detect low-confidence blobs, and autonomously diving into L2 (maximum zoom) only when a Region of Interest requires confident classification, before resetting to L0.

## 📂 Repository Structure
*   `predict.py`: The core FastAPI endpoint containing the multi-model YOLO inference, optical flow tracking logic, and camera sweep policy.
*   `run_multiscale.py`: OpenCV-based pipeline for multi-scale synthetic data generation and cascade training.
*   `api.py`: FastAPI server configuration.
*   `CHALLENGE_INSTRUCTIONS.md`: The original rules, constraints, and scoring metrics of the hackathon.

## 🚀 How to Run
We cant make the server run because the competition is finished, so the competition's live evaluation server is no longer active. This rep just showcase my architecture and my choices to resolve this competition. 
Generate the synthetic datasets and train the cascade models:
```bash
python run_multiscale.py