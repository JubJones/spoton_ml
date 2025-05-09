import os
import boto3
from botocore.exceptions import ClientError # For more specific error handling

# --- Configuration ---

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "https://s3.dagshub.com") # Default DagsHub S3 endpoint
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "YOUR_ACCESS_KEY_ID_HERE") # Replace or set env var
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "YOUR_SECRET_ACCESS_KEY_HERE") # Replace or set env var
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "spoton_ml") # Replace with your DagsHub repository name (e.g., "spoton_ml")


# This is the base directory on your local machine where videos will be downloaded.
# The script will create subdirectories like 'campus/c01/', 'factory/f01/' inside this.
LOCAL_DOWNLOAD_BASE_DIR = "downloaded_spoton_videos_boto3"

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

def download_videos_from_s3():
    """
    Downloads specified video sets from S3-compatible storage (like DagsHub's).
    """
    if "YOUR_ACCESS_KEY_ID_HERE" in AWS_ACCESS_KEY_ID or \
       "YOUR_SECRET_ACCESS_KEY_HERE" in AWS_SECRET_ACCESS_KEY or \
       "your_dagshub_repo_name" in S3_BUCKET_NAME:
        print("ERROR: Please configure S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and S3_BUCKET_NAME.")
        print("       It's best to set these as environment variables.")
        print("       You can find these details in your DagsHub repo: ... (Remote) -> S3 tab.")
        return

    print(f"Initializing Boto3 S3 client for endpoint: {S3_ENDPOINT_URL}, bucket: {S3_BUCKET_NAME}")
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        # Optional: Test connection by listing buckets (if permissions allow) or a specific prefix
        # s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, MaxKeys=1)
        # print("Successfully connected to S3.")
    except Exception as e:
        print(f"ERROR: Could not initialize Boto3 S3 client: {e}")
        print("Please check your S3 endpoint, credentials, and network connectivity.")
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

            # Construct the full remote key for the file in the S3 bucket
            remote_file_key = f"{remote_base_key}/{video_filename}"
            # Construct the full local path where the file will be saved
            local_file_path = os.path.join(local_camera_dir, video_filename)

            total_downloads_attempted += 1
            print(f"  Attempting to download: S3 Key='{remote_file_key}' from Bucket='{S3_BUCKET_NAME}'")
            print(f"  Saving to: '{local_file_path}'")

            # Check if file already exists locally to avoid re-downloading (optional)
            if os.path.exists(local_file_path):
                print(f"  INFO: File '{local_file_path}' already exists locally. Skipping download.")
                successful_downloads +=1 # Count as success if already present
                continue

            try:
                s3_client.download_file(
                    Bucket=S3_BUCKET_NAME,
                    Key=remote_file_key,
                    Filename=local_file_path,
                )
                print(f"  SUCCESS: Downloaded '{remote_file_key}' to '{local_file_path}'")
                successful_downloads += 1
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code')
                if error_code == '404' or error_code == 'NoSuchKey':
                    print(f"  ERROR: File not found in S3 bucket: '{remote_file_key}'. Please check the path and bucket.")
                elif error_code == '403' or 'Forbidden' in str(e) or 'AccessDenied' in str(e):
                    print(f"  ERROR: Access Denied for S3 key '{remote_file_key}'. Check your S3 credentials and permissions.")
                else:
                    print(f"  ERROR: An S3 client error occurred while downloading '{remote_file_key}': {e}")
            except Exception as e:
                print(f"  ERROR: An unexpected error occurred while downloading '{remote_file_key}': {e}")

    print(f"\n--- Download Summary ---")
    print(f"Total download attempts: {total_downloads_attempted}")
    print(f"Successful downloads (or already existed): {successful_downloads}")
    print(f"Failed downloads: {total_downloads_attempted - successful_downloads}")
    print(f"All downloaded files are located under: {os.path.abspath(LOCAL_DOWNLOAD_BASE_DIR)}")

if __name__ == "__main__":
    # Initial check for placeholder values is now inside the function
    if not VIDEO_SETS_TO_DOWNLOAD:
        print("INFO: `VIDEO_SETS_TO_DOWNLOAD` is empty. No videos will be downloaded.")
    else:
        download_videos_from_s3()