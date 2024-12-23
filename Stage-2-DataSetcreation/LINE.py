
# line_mask_generator.py
import logging
from pathlib import Path
from common import read_zip, generate_line_mask, get_masked, TARGET_SHAPE, IN_DIR, S3_BUCKET_NAME, S3_MAIN_DIR, upload_image_to_s3, s3_client

# S3 Configuration
S3_SUB_DIR = "retrain_data/line/label"

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
        upload_image_to_s3(masked_img, S3_BUCKET_NAME, s3_key)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s;%(levelname)s;%(message)s')

    try:
        # List ZIP files in the S3 bucket
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix =f"{S3_MAIN_DIR}/{IN_DIR}")
        for obj in response.get('Contents', []):
            s3_key = obj['Key']
            if s3_key.endswith('.zip'): # Process only ZIP files
                logging.info(f"Processing ZIP files from S3: {s3_key}")
                read_zip(S3_BUCKET_NAME, s3_key, generate_lines_from_json)
    except Exception as e:
        logging.error(f"Error listing or processing ZIP files from S3: {e}")

