import sys
import cv2
import numpy as np
import supervision as sv

def detect_moving_objects(fg_mask, min_area):
    '''
    Detect moving objects in the foreground mask and return their bounding boxes.

    Parameters:
    - fg_mask: np.ndarray, binary mask of the foreground.
    - min_area: int, minimum area of detected objects.

    Returns:
    - bboxes: np.ndarray of shape (N, 4), where N is the number of detected objects, 
      and each row is [x1, y1, x2, y2].
    '''

    # Apply morphological operations to clean up the foreground mask.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_DILATE, kernel, iterations=2)
    
    # Find contours in the cleaned foreground mask and filter them based on area to get bounding boxes.
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append([x, y, x + w, y + h])
    return np.array(bboxes)

def analize_video(video_path, min_area=100, min_frames=30, var_threshold=12, history=500, verbose=False, skip_corrupted_frames=0):
    '''
    Compute the detection on a video using background subtraction.

    Parameters:
    - video_path: str, path to the video file.
    - min_area: int, minimum area of detected objects.
    - min_frames: int, minimum number of frames an object must be detected for.
    - var_threshold: int, threshold for variance in background subtraction.
    - history: int, number of frames to consider for background model.
    - verbose: bool, whether to print progress information.
    - skip_corrupted_frames: int, maximum number of corrupted frames to skip.

    Returns:
    - up_count: int, number of objects that crossed the line upward.
    - down_count: int, number of objects that crossed the line downward.
    - detections: dict, mapping of tracker_id to a dict of frame_idx to bounding box coordinates.
    '''
    if verbose:
        print(f"Analizando video: {video_path}")

    # Video openning and parameters extraction
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames_minus_one = num_frames - 1

    # Background subtractor and tracker initialization
    backSub = cv2.createBackgroundSubtractorMOG2(history, var_threshold, detectShadows=True)
    tracker = sv.ByteTrack()
    
    # Initial detections dictionary to store preliminary detections before filtering
    preliminar_detections = {}
    
    if verbose:
        print("Registrando movimientos...")
    
    # Parameters for handling corrupted frames
    max_skips = skip_corrupted_frames
    RED = "\033[31m"
    RESET = "\033[0m"

    frame_idx = 0
    while frame_idx < num_frames:
        ret, frame = cap.read()
        # If the frame is not read correctly, it might be corrupted. We will try to skip it and continue.
        if not ret:
            # For good console output
            if max_skips == skip_corrupted_frames:
                print("")

            print(f"{RED}Frame {frame_idx} corrupto. Saltando…{RESET}", file=sys.stderr)

            # Skip frame
            frame_idx += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            # Abort if we have skipped too many frames in a row
            max_skips -= 1
            if max_skips == 0:
                print(f"{RED}Demasiados frames corruptos seguidos. Abortando.{RESET}", file=sys.stderr)
                break

            continue

        # The frame was read successfully, we can reset the skip counter
        max_skips = skip_corrupted_frames

        if verbose:
            print(f"\rProcesando frame {frame_idx}/{num_frames_minus_one}", end="", flush=True)
        
        # ------------------------------
        # ---- DETECTION Y TRACKING ----
        # ------------------------------

        # Apply background subtraction and detect moving objects
        fg_mask = backSub.apply(frame)
        bboxes = detect_moving_objects(fg_mask, min_area)

        # If there are detections, we create a Detections object and update the tracker.
        if len(bboxes) > 0:
            detections = sv.Detections(
                xyxy=bboxes.astype(np.float32),
                confidence=np.ones(len(bboxes), dtype=np.float32),
                class_id=np.zeros(len(bboxes), dtype=int)
            )
            detections = tracker.update_with_detections(detections)

            # Preliminar detections are stored in a dictionary with tracker_id as key and another dictionary as value,
            # where the key is the frame index and the value is the bounding box coordinates.
            for i in range(len(detections)):
                tid = int(detections.tracker_id[i])
                coords = detections.xyxy[i].tolist()
                
                if tid not in preliminar_detections:
                    preliminar_detections[tid] = {}
                preliminar_detections[tid][frame_idx] = coords
        frame_idx += 1
    cap.release()
    
    # ------------------------------
    # --------- FILTERING ----------
    # ------------------------------

    # Application of time-based filtering.
    detections = {tid: d for tid, d in preliminar_detections.items() if len(d) >= min_frames}
    
    # ------------------------------
    # ----- LINE ZONE COUNTING -----
    # ------------------------------
    if verbose:
        print(f"\nContabilizando flujo:")
    
    # LineZone initialization.
    line_start = sv.Point(0, int(h * 0.5))
    line_end = sv.Point(w, int(h * 0.5))
    line_zone = sv.LineZone(start=line_start, end=line_end)

    # Iteration for each frame.
    frame_idx = 0
    for _ in range(num_frames):
        if verbose:
            print(f"\rProcesando frame {frame_idx}/{num_frames_minus_one}", end="", flush=True)

        current_xyxy = []
        current_ids = []
        
        # Obtention of the detections for the current frame.
        for tid, frames_dict in detections.items():
            if frame_idx in frames_dict:
                current_xyxy.append(frames_dict[frame_idx])
                current_ids.append(tid)

        if current_xyxy:
            # Creation of a Detections object with the detections of the current frame.
            real_detections = sv.Detections(
                xyxy=np.array(current_xyxy, dtype=np.float32),
                tracker_id=np.array(current_ids, dtype=int),
                class_id=np.zeros(len(current_ids), dtype=int)
            )

            # Trigger activation for counting
            line_zone.trigger(real_detections)

        frame_idx += 1

    if verbose:
        print(f"\nProcesamiento completado del video: {video_path}")
    
    # Return the counts of objects that crossed the line in both directions and the final detections 
    # dictionary after filtering.
    return line_zone.in_count, line_zone.out_count, detections