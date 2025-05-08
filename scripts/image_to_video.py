import cv2
import os
import glob
import math

# --- Configuration ---
FPS = 23.0
NUM_SUB_VIDEOS = 4
OUTPUT_BASE_DIR = "output_sub_videos"  # Base directory where sub-videos will be saved

# IMPORTANT: Configure this list with your camera data
# 'env': A name for the environment (e.g., 'campus', 'factory'). This will be a subfolder in OUTPUT_BASE_DIR.
# 'cam_id': A unique camera identifier (e.g., 'c01', 'f01'). This will be another subfolder.
# 'rgb_path': The absolute or relative path to the directory containing the .jpg image sequence for this camera.
CAMERAS_TO_PROCESS = [
    # Example for Campus Cameras (assuming MTMMC 'train' set structure for scenes s01-s04)
    # You'll need to map your 'campus' and 'factory' logical cameras to specific SXX/cYY paths from MTMMC
    {"env": "campus", "cam_id": "c01", "rgb_path": "/Volumes/HDD/MTMMC/val/val/s37/c01/rgb"},
    {"env": "campus", "cam_id": "c02", "rgb_path": "/Volumes/HDD/MTMMC/val/val/s37/c02/rgb"}, # Adjust scene/camera as needed
    {"env": "campus", "cam_id": "c03", "rgb_path": "/Volumes/HDD/MTMMC/val/val/s37/c03/rgb"}, # Adjust scene/camera as needed
    {"env": "campus", "cam_id": "c04", "rgb_path": "/Volumes/HDD/MTMMC/val/val/s37/c05/rgb"}, # Adjust scene/camera as needed

    # Example for Factory Cameras (assuming MTMMC 'val' set structure for scenes s10-s13 or similar)
    {"env": "factory", "cam_id": "f01", "rgb_path": "/Volumes/HDD/MTMMC/val/val/s14/c09/rgb"}, # Adjust scene/camera as needed
    {"env": "factory", "cam_id": "f02", "rgb_path": "/Volumes/HDD/MTMMC/val/val/s14/c12/rgb"}, # Adjust scene/camera as needed
    {"env": "factory", "cam_id": "f03", "rgb_path": "/Volumes/HDD/MTMMC/val/val/s14/c13/rgb"}, # Adjust scene/camera as needed
    {"env": "factory", "cam_id": "f04", "rgb_path": "/Volumes/HDD/MTMMC/val/val/s14/c16/rgb"}, # Adjust scene/camera as needed
    # Add more cameras if you have them, or adjust the paths above.
]

# Ensure the main output directory exists
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def create_sub_videos_for_camera(env_name, camera_id, images_dir_path, output_root_dir):
    """
    Processes an image sequence from a specific camera and creates sub-videos.
    """
    print(f"\nProcessing camera: Environment='{env_name}', CameraID='{camera_id}'")
    print(f"Image source: {images_dir_path}")

    image_files = sorted(glob.glob(os.path.join(images_dir_path, '*.jpg'))) # Assuming JPG images

    if not image_files:
        print(f"WARNING: No image files found in {images_dir_path}. Skipping this camera.")
        return

    total_frames = len(image_files)
    print(f"Found {total_frames} frames.")

    if total_frames == 0:
        print(f"WARNING: Zero frames found for {camera_id}. Skipping.")
        return

    # Determine frame dimensions from the first image
    try:
        first_image = cv2.imread(image_files[0])
        if first_image is None:
            print(f"ERROR: Could not read the first image: {image_files[0]}. Skipping this camera.")
            return
        height, width, layers = first_image.shape
        frame_size = (width, height)
    except Exception as e:
        print(f"ERROR: Could not get frame dimensions from {image_files[0]}: {e}. Skipping this camera.")
        return

    frames_per_sub_video = math.ceil(total_frames / NUM_SUB_VIDEOS)
    if frames_per_sub_video == 0 : # Should not happen if total_frames > 0
        print(f"ERROR: Calculated zero frames per sub-video for {camera_id}. This usually means no frames were found. Skipping.")
        return

    print(f"Each of the {NUM_SUB_VIDEOS} sub-videos will have up to {frames_per_sub_video} frames.")

    # Create output directory for this specific camera
    camera_output_dir = os.path.join(output_root_dir, env_name, camera_id)
    os.makedirs(camera_output_dir, exist_ok=True)

    for i in range(NUM_SUB_VIDEOS):
        sub_video_number = i + 1
        output_video_filename = f"sub_video_{sub_video_number:02d}.mp4"
        output_video_path = os.path.join(camera_output_dir, output_video_filename)

        start_index = i * frames_per_sub_video
        end_index = min((i + 1) * frames_per_sub_video, total_frames)

        # If start_index is beyond the total number of frames, we've created enough sub-videos
        if start_index >= total_frames:
            break

        print(f"  Creating {output_video_filename} (frames {start_index} to {end_index-1})...")

        # Define the codec and create VideoWriter object
        # Using 'mp4v' for MP4. Other options include 'XVID' for AVI.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, FPS, frame_size)

        if not video_writer.isOpened():
            print(f"ERROR: Could not open video writer for {output_video_path}. Check OpenCV installation and codecs.")
            continue # Try next sub-video or camera

        for frame_idx in range(start_index, end_index):
            image_path = image_files[frame_idx]
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"    WARNING: Could not read frame {image_path}. Skipping this frame.")
                continue
            
            # Ensure frame is the correct size (can happen if dataset has inconsistent image sizes)
            if frame.shape[0] != height or frame.shape[1] != width:
                print(f"    WARNING: Frame {image_path} has dimensions {frame.shape[:2]} but expected {frame_size}. Resizing.")
                frame = cv2.resize(frame, frame_size)

            video_writer.write(frame)

        video_writer.release()
        print(f"  Successfully created {output_video_path}")

    print(f"Finished processing for camera: {camera_id}")


if __name__ == "__main__":
    if not CAMERAS_TO_PROCESS:
        print("Configuration error: `CAMERAS_TO_PROCESS` list is empty. Please define camera paths.")
    else:
        for camera_info in CAMERAS_TO_PROCESS:
            if not all(k in camera_info for k in ("env", "cam_id", "rgb_path")):
                print(f"Skipping invalid camera entry: {camera_info}. Missing 'env', 'cam_id', or 'rgb_path'.")
                continue
            create_sub_videos_for_camera(
                camera_info["env"],
                camera_info["cam_id"],
                camera_info["rgb_path"],
                OUTPUT_BASE_DIR
            )
        print("\nAll configured cameras processed.")