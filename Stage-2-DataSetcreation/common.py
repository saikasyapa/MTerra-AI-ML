# common.py
import cv2
import json
import zipfile
import logging
import numpy as np
import boto3
from io import BytesIO

TARGET_SHAPE = (1664, 1024)
THICKNESS = 8

S3_BUCKET_NAME = "kadaster-magnasoft"
S3_MAIN_DIR = "Kadaster-AI-ML"
IN_DIR = "vector-data"


s3_client = boto3.client('s3')

def upload_image_to_s3(image, s3_bucket, s3_key):
    """
    Upload an image (in memory) directly to S3 as a PNG.
    """
    try:
        # Encode image to PNG format in memory
        _, buffer = cv2.imencode('.png', image)
        s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=BytesIO(buffer), ContentType='image/png')
        logging.info(f"Uploaded to S3: s3://{s3_bucket}/{s3_key}")
    except Exception as e:
        logging.error(f"Error uploading image to S3: {e}")

def generate_line_mask(lines, mask_shape):
    mask = np.zeros(mask_shape, dtype=np.uint8)
    for line in lines:
        cv2.line(mask, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), 255,
                 thickness=THICKNESS, lineType=cv2.LINE_8)
    mask = cv2.resize(mask, (TARGET_SHAPE[1], TARGET_SHAPE[0])).astype(np.uint8)
    return np.nonzero(mask) if np.any(mask) else None


def get_masked(line_masks, building_masks, border_masks, mask_shape):
    mask_img = np.zeros(mask_shape, dtype=np.uint8)
    if line_masks is not None:
        mask_img[line_masks[0], line_masks[1]] = 255
    if building_masks is not None:
        mask_img[building_masks[0], building_masks[1]] = 255
    if border_masks is not None:
        mask_img[border_masks[0], border_masks[1]] = 255
    return mask_img


def read_zip(s3_bucket,s3_key, process_fn):
    try:
        # Download ZIP file from s3 to a temporary in-memory buffer
        zip_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        with zipfile.ZipFile(BytesIO(zip_obj['Body'].read()), 'r') as archive:
            prefix, postfix = 'observations/snapshots/latest/', '.latest.json'
            sketch_files = [x for x in archive.namelist() if x.startswith(prefix) and x.endswith(postfix)]
            for i, sketch_file in enumerate(sketch_files):
                sketch_name = sketch_file[len(prefix):-len(postfix)]
                logging.info(f'Processing sketch: {i}: {sketch_name}.')

                with archive.open(sketch_file, 'r') as afh:
                    json_data = json.loads(afh.read())

                for attachment, details in json_data['attachments'].items():
                    try:
                        dimensions = details['properties']['dimensions']
                        height, width = dimensions[1], dimensions[0]
                        image_shape = (height, width)
                    except KeyError:
                        logging.warning(f"Missing 'dimensions' key for attachment: {attachment}. Skipping...")
                        continue

                    # Fix argument order
                    process_fn(json_data, sketch_name, image_shape, attachment)
    except Exception as e:
        logging.error(f"Error processing ZIP file from S3: {e}")
