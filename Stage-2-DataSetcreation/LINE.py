
# line_mask_generator.py
import logging
import os.path
from common import (read_zip, generate_line_mask, get_masked, TARGET_SHAPE, IN_DIR, S3_BUCKET_NAME, S3_MAIN_DIR,
                    upload_image_to_s3, s3_client, S3_LOG_DIR)
from tempfile import gettempdir

# S3 Configuration
S3_SUB_DIR = "retrain_data/line/label"


# Get a cross-platform temporary directory and define lof file path
temp_dir = gettempdir() # This dynamically retrieves the temp directory (e.g., /tmp on Linux, C:\Temp on Windows)
log_file_path = os.path.join(temp_dir, "line_mask_generator.log")

# Ensure the directory for the log file exists
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
# Set up logging to a local file
file_handler = logging.FileHandler(log_file_path)
stream_handler = logging.StreamHandler()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s;%(levelname)s;%(message)s',
    handlers=[file_handler, stream_handler]
)

def generate_lines_from_json(obs, sketch_name, image_shape, attachment):
    lines, points_dict = obs['lines'], obs['points']
    lines = {k: v for k, v in lines.items() if v.get('attachment') == attachment}
    segments = []
    for line, details in lines.items():
        points = details['points']
        for start, stop in zip(points, points[1:]):
            segments.append([points_dict[start]['position'], points_dict[stop]['position']])
    line_masks = generate_line_mask(segments, image_shape)
    if line_masks:
        masked_img = get_masked(line_masks, None, None, TARGET_SHAPE)
        s3_key = f"{S3_MAIN_DIR}/{S3_SUB_DIR}/{sketch_name}.{attachment}.png"

        # Upload to S3
        try:
            upload_image_to_s3(masked_img, S3_BUCKET_NAME, s3_key)
            logging.info(f"Successfully uploaded {sketch_name}.{attachment}.png to {S3_BUCKET_NAME}")
        except Exception as exc:
            logging.error(f"failed to upload {sketch_name}.{attachment}.png to {S3_BUCKET_NAME}: {exc}")

    else:
        logging.warning(f"No line mask generated for {sketch_name}.{attachment}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s;%(levelname)s;%(message)s')

    try:
        # List ZIP files in the S3 bucket
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix =f"{S3_MAIN_DIR}/{IN_DIR}")
        if 'Contents' not in response:
            logging.error(f"No contents found in the {S3_MAIN_DIR}")
        else:
            for obj in response.get('Contents', []):
                s3_key = obj['Key']
                if s3_key.endswith('.zip'): # Process only ZIP files
                    logging.info(f"Processing ZIP files from S3: {s3_key}")
                    try:
                        read_zip(S3_BUCKET_NAME, s3_key, generate_lines_from_json)
                        logging.info(f"Successfully processed ZIP file: {s3_key}")
                    except Exception as e:
                        logging.error(f"Failed to process zip file {s3_key}: {e}")
    except Exception as e:
        logging.error(f"Error listing or processing ZIP files from S3: {e}")

    # Upload the log file to S3 after processing is done
    try:
        s3_log_key=f"{S3_MAIN_DIR}/{S3_LOG_DIR}/line_mask_generator.txt"
        logging.info(f"Uploading log file to S3: {s3_log_key}")
        
        # Upload the log file to S3
        with open(log_file_path, 'rb') as log_file:
            s3_client.put_object(Body=log_file, Bucket=S3_BUCKET_NAME, Key=s3_log_key)
        logging.info("log file uploaded successfully.")
    
    except Exception as e:
        logging.error(f"Error uploading log file to S3: {e}")
    
    finally:
        # Close the logging file handler
        file_handler.close()
        logging.getLogger().removeHandler(file_handler)
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
            logging.info("Cleaned up the local log file.")


