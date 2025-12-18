import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os
import subprocess
from PIL import Image, ImageDraw, ImageFont

def get_hips_center(keypoints, conf_threshold=0.3):
    """
    Advanced hip center estimation using torso triangulation.
    Handles bent bodies and missing hip points by projecting from shoulders.
    """
    # Keypoints: Nose(0), L-Shoulder(5), R-Shoulder(6), L-Hip(11), R-Hip(12)
    nose = keypoints[0]
    l_sh = keypoints[5]
    r_sh = keypoints[6]
    l_hip = keypoints[11]
    r_hip = keypoints[12]

    # 1. Ideal case: Both hips are visible
    if l_hip[2] > conf_threshold and r_hip[2] > conf_threshold:
        center_x = (l_hip[0] + r_hip[0]) / 2
        center_y = (l_hip[1] + r_hip[1]) / 2
        
        # Add a small downward offset (roughly 50% of shoulder width) to be "below the center"
        if l_sh[2] > conf_threshold and r_sh[2] > conf_threshold:
            sh_width = np.linalg.norm(l_sh[:2] - r_sh[:2])
            center_y += sh_width * 0.50
            
        return (int(center_x), int(center_y))

    # 2. Fallback: Project from shoulders if hips are obscured (e.g. immersed)
    if l_sh[2] > conf_threshold and r_sh[2] > conf_threshold:
        sh_mid = np.array([(l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2])
        
        # Use Nose to find the "up" vector of the torso
        if nose[2] > conf_threshold:
            # Vector from nose to shoulder midpoint defines the spine direction
            spine_vec = sh_mid - nose[:2]
            # Project down from shoulders (torso is roughly 2.1x the head-to-shoulder distance)
            # Increased to 2.1x to lower the position as requested
            hip_est = sh_mid + spine_vec * 2.1
            return (int(hip_est[0]), int(hip_est[1]))
        
        # If nose is missing, just use the visible hip if available
        if l_hip[2] > conf_threshold: return (int(l_hip[0]), int(l_hip[1]))
        if r_hip[2] > conf_threshold: return (int(r_hip[0]), int(r_hip[1]))

    return None

def is_immersed(keypoints, conf_threshold=0.2):
    """
    Refined immersion detection. 
    Returns True if the lower extremities are significantly less visible than the upper body.
    Extremely sensitive to catch immersion in transparent water or low light.
    """
    # Keypoints: Nose(0), L-Shoulder(5), R-Shoulder(6), L-Hip(11), R-Hip(12), 
    # L-Knee(13), R-Knee(14), L-Ankle(15), R-Ankle(16)
    
    # Upper body: Nose and Shoulders (most likely to be above water)
    upper_points = [keypoints[0], keypoints[5], keypoints[6]]
    upper_conf = sum(p[2] for p in upper_points) / 3
    
    # Mid body: Hips
    hip_conf = (keypoints[11][2] + keypoints[12][2]) / 2
    
    # Lower body: Knees and Ankles
    lower_points = [keypoints[13], keypoints[14], keypoints[15], keypoints[16]]
    lower_conf = sum(p[2] for p in lower_points) / 4
    
    # If upper body is even slightly visible (low light threshold)
    if upper_conf > 0.15:
        # 1. Ultra-aggressive relative check for hips. 
        # In very transparent water, the confidence drop is minimal.
        # If hips are even 3% less confident than the head/shoulders, 
        # we assume they are underwater or distorted by the surface.
        if hip_conf < (upper_conf * 0.97):
            return True
            
        # 2. Lower body check. If knees/ankles are 15% less confident than upper body,
        # it indicates the person is deep enough that the lower half is obscured.
        if lower_conf < (upper_conf * 0.85):
            return True
            
    # 3. Absolute fallback: if lower body is low confidence while upper is somewhat visible
    if upper_conf > 0.2 and lower_conf < 0.35:
        return True
        
    # 4. If hips are significantly lower in the frame than shoulders but have lower confidence,
    # it's a strong indicator of water refraction/immersion.
    y_upper = (keypoints[5][1] + keypoints[6][1]) / 2
    y_hip = (keypoints[11][1] + keypoints[12][1]) / 2
    if y_hip > y_upper and hip_conf < upper_conf:
        return True

    return False

def draw_emoji(frame, text, position, size=50):
    """
    Draw an emoji on an OpenCV frame using PIL.
    """
    # Convert OpenCV BGR to PIL RGB
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Try to load a font that supports emojis
    try:
        # Windows emoji font
        font = ImageFont.truetype("seguiemj.ttf", size)
    except:
        try:
            # Linux/macOS fallback
            font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", size)
        except:
            # Default font if no emoji font found
            font = ImageFont.load_default()

    # Calculate text position to center it
    # Use textbbox for newer PIL versions
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Fallback for older PIL
        w, h = draw.textsize(text, font=font)
        
    draw.text((position[0] - w//2, position[1] - h//2), text, font=font, embedded_color=True)
    
    # Convert back to OpenCV BGR
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    parser = argparse.ArgumentParser(description="Skeleton Tracking with YOLOv8 Pose")
    parser.add_argument("--input", type=str, default="assets/piscine.mp4", help="Path to input video file")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output video file")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt", help="YOLOv8 pose model to use")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for keypoints")
    parser.add_argument("--no-show", action="store_true", help="Do not display the tracking window")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="Manual rotation in degrees (clockwise)")
    
    args = parser.parse_args()

    # Load YOLOv8 Pose model
    model = YOLO(args.model)

    # Open input video
    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print(f"Error: Could not open video {args.input}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0 # Fallback
    
    # Check for rotation metadata (if supported by backend)
    meta_rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    if meta_rotation is None: meta_rotation = 0
    print(f"Metadata rotation: {meta_rotation}")

    # Read first frame to get actual dimensions
    first_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame is not None:
            # Apply manual or meta rotation to the first frame to get correct dimensions
            final_rotate = args.rotate if args.rotate != 0 else meta_rotation
            
            if final_rotate == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif final_rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif final_rotate == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
            first_frame = frame
            break
            
    if first_frame is None:
        print("Error: Could not read any valid frame.")
        cap.release()
        return
    
    height, width = first_frame.shape[:2]
    # Ensure dimensions are even (required by some codecs)
    width = (width // 2) * 2
    height = (height // 2) * 2

    # Setup VideoWriter with fallback mechanism
    temp_output = "temp_no_audio.mp4"
    codecs = ['avc1', 'mp4v']
    out = None
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"Using codec: {codec}")
            break
        else:
            print(f"Codec {codec} failed, trying next...")
    
    if out is None or not out.isOpened():
        print("Error: Could not initialize VideoWriter with any codec.")
        cap.release()
        return

    print(f"Processing video: {args.input}")
    print(f"Resolution: {width}x{height} @ {fps} FPS")

    # Persistence tracker state with physics
    tracker_state = {} # {track_id: {'pos': [x, y], 'vel': [vx, vy], 'prev_kpts': None, 'cooldown': int}}
    COOLDOWN_FRAMES = 5

    try:
        # Process the first frame we already read
        frame = first_frame
        while True:
            # Ensure frame matches VideoWriter dimensions
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            # Run YOLOv8 tracking
            results = model.track(frame, persist=True, verbose=False)
            
            current_frame_ids = set()

            if results and results[0].keypoints is not None:
                keypoints_data = results[0].keypoints.data.cpu().numpy()
                # Get tracking IDs
                track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [None] * len(keypoints_data)
                
                for person_keypoints, track_id in zip(keypoints_data, track_ids):
                    hips_center = get_hips_center(person_keypoints, args.conf)
                    
                    if track_id is not None:
                        if track_id not in tracker_state:
                            tracker_state[track_id] = {'pos': None, 'vel': [0, 0], 'prev_kpts': None, 'cooldown': 0}
                        
                        # Update movement vector using average displacement of visible keypoints
                        if tracker_state[track_id]['prev_kpts'] is not None:
                            prev_kpts = tracker_state[track_id]['prev_kpts']
                            displacements = []
                            for i in range(len(person_keypoints)):
                                if person_keypoints[i][2] > args.conf and prev_kpts[i][2] > args.conf:
                                    displacements.append(person_keypoints[i][:2] - prev_kpts[i][:2])
                            
                            if displacements:
                                avg_disp = np.mean(displacements, axis=0)
                                # Smooth the velocity vector (EMA)
                                alpha = 0.6
                                tracker_state[track_id]['vel'] = [
                                    alpha * avg_disp[0] + (1 - alpha) * tracker_state[track_id]['vel'][0],
                                    alpha * avg_disp[1] + (1 - alpha) * tracker_state[track_id]['vel'][1]
                                ]
                        
                        tracker_state[track_id]['prev_kpts'] = person_keypoints.copy()

                    # Draw maple leaf emoji at Hips Center ONLY if NOT immersed
                    if hips_center and not is_immersed(person_keypoints, args.conf):
                        if track_id is not None:
                            # Prevent big jumps by blending detection with prediction (EMA)
                            if tracker_state[track_id]['pos'] is not None:
                                prev_pos = np.array(tracker_state[track_id]['pos'])
                                vel = np.array(tracker_state[track_id]['vel'])
                                predicted_pos = prev_pos + vel
                                detected_pos = np.array(hips_center)
                                
                                # Blend detection with prediction (EMA)
                                # This naturally smooths out jitter and prevents sudden jumps
                                pos_alpha = 0.35 
                                smoothed_pos = pos_alpha * detected_pos + (1 - pos_alpha) * predicted_pos
                                
                                # Hard clamp: Limit the maximum deviation from the predicted path
                                diff = smoothed_pos - predicted_pos
                                dist = np.linalg.norm(diff)
                                # Max jump allowed is 15 pixels or 1.5x the current velocity magnitude
                                max_jump = max(15.0, np.linalg.norm(vel) * 1.5)
                                
                                if dist > max_jump:
                                    smoothed_pos = predicted_pos + (diff / dist) * max_jump
                                
                                tracker_state[track_id]['pos'] = list(smoothed_pos)
                            else:
                                tracker_state[track_id]['pos'] = list(hips_center)
                                
                            tracker_state[track_id]['cooldown'] = COOLDOWN_FRAMES
                            current_frame_ids.add(track_id)
                            
                            # Calculate dynamic size based on body scale (shoulder width)
                            emoji_size = 60
                            l_sh_kpt = person_keypoints[5]
                            r_sh_kpt = person_keypoints[6]
                            if l_sh_kpt[2] > args.conf and r_sh_kpt[2] > args.conf:
                                sh_width = np.linalg.norm(l_sh_kpt[:2] - r_sh_kpt[:2])
                                # Base size 60 for a typical shoulder width of ~100 pixels
                                emoji_size = int(max(30, min(120, sh_width * 0.8)))
                            
                            draw_pos = (int(tracker_state[track_id]['pos'][0]), int(tracker_state[track_id]['pos'][1]))
                            frame = draw_emoji(frame, "🍁", draw_pos, size=emoji_size)
                        else:
                            frame = draw_emoji(frame, "🍁", hips_center, size=60)

            # Handle persistence with physics (Momentum)
            ids_to_remove = []
            for track_id, state in tracker_state.items():
                if track_id not in current_frame_ids:
                    if state['cooldown'] > 0 and state['pos'] is not None:
                        # Apply physics: Update position by velocity
                        state['pos'][0] += state['vel'][0]
                        state['pos'][1] += state['vel'][1]
                        
                        # Draw at the predicted physical position
                        draw_pos = (int(state['pos'][0]), int(state['pos'][1]))
                        
                        # Calculate dynamic size for persistent frames
                        emoji_size = 60
                        if state['prev_kpts'] is not None:
                            pk = state['prev_kpts']
                            if pk[5][2] > 0.3 and pk[6][2] > 0.3:
                                sh_width = np.linalg.norm(pk[5][:2] - pk[6][:2])
                                emoji_size = int(max(30, min(120, sh_width * 0.8)))
                        
                        # Only draw if within frame boundaries
                        if 0 <= draw_pos[0] < width and 0 <= draw_pos[1] < height:
                            frame = draw_emoji(frame, "🍁", draw_pos, size=emoji_size)
                        
                        state['cooldown'] -= 1
                    else:
                        ids_to_remove.append(track_id)
            
            for track_id in ids_to_remove:
                del tracker_state[track_id]

            # Write frame to output
            out.write(frame)

            # Display frame
            if not args.no_show:
                cv2.imshow('Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Read next frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply rotation to subsequent frames
            final_rotate = args.rotate if args.rotate != 0 else meta_rotation
            if final_rotate == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif final_rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif final_rotate == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Merge audio and compress using ffmpeg
        print("Merging audio and compressing...")
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_output,
                '-i', args.input,
                '-map', '0:v:0',
                '-map', '1:a:0?',
                '-vf', 'scale=-2:720',
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'medium',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-shortest',
                args.output
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Processing complete. Saved to {args.output}")
            
            # Clean up temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)
        except subprocess.CalledProcessError as e:
            print(f"Error merging audio: {e.stderr.decode()}")
            print(f"Video saved without audio at {temp_output}")
        except Exception as e:
            print(f"An unexpected error occurred during audio merge: {e}")
            print(f"Video saved without audio at {temp_output}")

if __name__ == "__main__":
    main()

