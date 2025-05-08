import os
from dagshub import get_repo_bucket_client
from botocore.exceptions import ClientError # For more specific error handling

# --- Configuration ---
DAGSHUB_USER = "Jwizzed"  # Replace with your DagsHub username
DAGSHUB_REPO_NAME_ONLY = "spoton_ml"  # Replace with your DagsHub repository name (e.g., "spoton_data" or the repo where videos are stored)

# This is the base directory on your local machine where videos will be downloaded.
# The script will create subdirectories like 'campus/c01/', 'factory/f01/' inside this.
LOCAL_DOWNLOAD_BASE_DIR = "downloaded_spoton_videos"

VIDEO_SETS_TO_DOWNLOAD = [
    # Campus Cameras
    {"remote_base_key": "video_s37/c01", "env": "campus", "cam_id": "c01", "num_sub_videos": 4},
    {"remote_base_key": "video_s37/c02", "env": "campus", "cam_id": "c02", "num_sub_videos": 4},
    {"remote_base_key": "video_s37/c03", "env": "campus", "cam_id": "c03", "num_sub_videos": 4},
    {"remote_base_key": "video_s37/c05", "env": "campus", "cam_id": "c05", "num_sub_videos": 4},

    # Factory Cameras
    {"remote_base_key": "video_s14/c09", "env": "factory", "cam_id": "c09", "num_sub_videos": 4},
    {"remote_base_key": "video_s14/c12", "env": "factory", "cam_id": "c12", "num_sub_videos": 4},
    {"remote_base_key": "video_s14/c13", "env": "factory", "cam_id": "c13", "num_sub_videos": 4},
    {"remote_base_key": "video_s14/c16", "env": "factory", "cam_id": "c16", "num_sub_videos": 4},
    # Add or modify entries above to match your DagsHub structure and desired downloads.
]

def download_videos_from_dagshub():
    """
    Downloads specified video sets from DagsHub.
    """
    print(f"Initializing DagsHub client for repo: {DAGSHUB_USER}/{DAGSHUB_REPO_NAME_ONLY}")
    try:
        boto_client = get_repo_bucket_client(
            f"{DAGSHUB_USER}/{DAGSHUB_REPO_NAME_ONLY}",
            flavor="boto"
        )
    except Exception as e:
        print(f"ERROR: Could not initialize DagsHub client: {e}")
        print("Please ensure you have run 'dagshub login' and the repository details are correct.")
        return

    print(f"Base local download directory: {os.path.abspath(LOCAL_DOWNLOAD_BASE_DIR)}")
    os.makedirs(LOCAL_DOWNLOAD_BASE_DIR, exist_ok=True)

    total_downloads_attempted = 0
    successful_downloads = 0

    for video_set in VIDEO_SETS_TO_DOWNLOAD:
        env = video_set["env"]
        cam_id = video_set["cam_id"]
        remote_base_key = video_set["remote_base_key"]
        num_sub_videos = video_set["num_sub_videos"]

        # Create local directory structure for this camera set
        local_camera_dir = os.path.join(LOCAL_DOWNLOAD_BASE_DIR, env, cam_id)
        os.makedirs(local_camera_dir, exist_ok=True)
        print(f"\nProcessing: Environment='{env}', Camera='{cam_id}'")
        print(f"Local target directory: {local_camera_dir}")

        for i in range(num_sub_videos):
            sub_video_number = i + 1
            video_filename = f"sub_video_{sub_video_number:02d}.mp4"

            # Construct the full remote key for the file on DagsHub
            remote_file_key = f"{remote_base_key}/{video_filename}"
            # Construct the full local path where the file will be saved
            local_file_path = os.path.join(local_camera_dir, video_filename)

            total_downloads_attempted += 1
            print(f"  Attempting to download: Dagshub Key='{remote_file_key}'")
            print(f"  Saving to: '{local_file_path}'")

            # Check if file already exists locally to avoid re-downloading (optional)
            if os.path.exists(local_file_path):
                print(f"  INFO: File '{local_file_path}' already exists locally. Skipping download.")
                successful_downloads +=1 # Count as success if already present
                continue

            try:
                boto_client.download_file(
                    Bucket=DAGSHUB_REPO_NAME_ONLY,  # DagsHub repo name acts as the S3 bucket name
                    Key=remote_file_key,            # Path to the file within the DagsHub repo's storage
                    Filename=local_file_path,       # Local path to save the downloaded file
                )
                print(f"  SUCCESS: Downloaded '{remote_file_key}' to '{local_file_path}'")
                successful_downloads += 1
            except ClientError as e:
                if e.response['Error']['Code'] == '404' or e.response['Error']['Code'] == 'NoSuchKey':
                    print(f"  ERROR: File not found on DagsHub: '{remote_file_key}'. Please check the path and repo.")
                else:
                    print(f"  ERROR: An AWS S3 client error occurred while downloading '{remote_file_key}': {e}")
            except Exception as e:
                print(f"  ERROR: An unexpected error occurred while downloading '{remote_file_key}': {e}")

    print(f"\n--- Download Summary ---")
    print(f"Total download attempts: {total_downloads_attempted}")
    print(f"Successful downloads (or already existed): {successful_downloads}")
    print(f"Failed downloads: {total_downloads_attempted - successful_downloads}")
    print(f"All downloaded files are located under: {os.path.abspath(LOCAL_DOWNLOAD_BASE_DIR)}")

if __name__ == "__main__":
    if DAGSHUB_USER == "YourDagsHubUsername" or DAGSHUB_REPO_NAME_ONLY == "YourDagsHubRepoName":
        print("ERROR: Please configure `DAGSHUB_USER` and `DAGSHUB_REPO_NAME_ONLY` at the top of the script.")
    elif not VIDEO_SETS_TO_DOWNLOAD:
        print("INFO: `VIDEO_SETS_TO_DOWNLOAD` is empty. No videos will be downloaded.")
    else:
        download_videos_from_dagshub()