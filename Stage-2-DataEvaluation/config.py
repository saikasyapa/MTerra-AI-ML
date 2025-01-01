import os
import boto3
import logging
import numpy as np
import cv2

TARGET_SHAPE = (1664, 1024)
THICKNESS = 8

s3_client = boto3.client('s3')

S3_BUCKET_NAME = "kadaster-magnasoft"
S3_MAIN_DIR = "Kadaster-AI-ML"
IN_DIR = "vector-data"
OUT_DIR = "data_evaluation"
S3_LOG_DIR = "logs/dataevaluation"

line_sub_1 = "lines_GroundTruth"
building_sub_1 = "buildings_GroundTruth"
border_sub_1 ="borders_GroundTruth"

line_sub_2 = "lines_Detection"
building_sub_2 = "buildings_Detection"
border_sub_2 ="borders_Detection"

# Text Box path variables
textbox_ground = "textbox_ground_truth"
textbox_predicted = "textbox_predicted"

OUT_DIR_TEXT_BOX=f"{S3_MAIN_DIR}/{OUT_DIR}/{textbox_ground}/textbox_ground"
OUT_DIR_TEXT_BOX1=f"{S3_MAIN_DIR}/{OUT_DIR}/{textbox_predicted}/textbox_predicted"

# Subdirectories for Manual masks
lines_dir_1 = f"{S3_MAIN_DIR}/{OUT_DIR}/{line_sub_1}/lines_GroundTruth"
borders_dir_1 = f"{S3_MAIN_DIR}/{OUT_DIR}/{border_sub_1}/borders_GroundTruth"
buildings_dir_1 = f"{S3_MAIN_DIR}/{OUT_DIR}/{building_sub_1}/buildings_GroundTruth"


# Subdirectories for Detection masks
lines_dir_2 = f"{S3_MAIN_DIR}/{OUT_DIR}/{line_sub_2}/lines_Detection"
borders_dir_2 = f"{S3_MAIN_DIR}/{OUT_DIR}/{border_sub_2}/borders_Detection"
buildings_dir_2 = f"{S3_MAIN_DIR}/{OUT_DIR}/{building_sub_2}/buildings_Detection"

def save_mask_to_s3(bucket, prefix, filename, mask, image_shape):
    """Save the generated mask image to S3."""
    if mask is not None:
        masked_img = get_masked(mask, None, None, image_shape)
        _, buffer = cv2.imencode('.png', masked_img)
        s3_client.put_object(Bucket=bucket, Key=os.path.join(prefix, filename), Body=buffer.tobytes())
        logging.info(f"Saved mask to S3: s3://{bucket}/{os.path.join(prefix, filename)}")

def generate_line_mask(segments, mask_shape):
    """Generate a mask from given line segments."""
    mask = np.zeros(mask_shape, dtype=np.uint8)

    if not segments:
        logging.warning("No segments found, returning empty mask.")
        return None

    for segment in segments:
        start, end = segment
        if (0 <= start[0] < mask_shape[1] and 0 <= start[1] < mask_shape[0] and
                0 <= end[0] < mask_shape[1] and 0 <= end[1] < mask_shape[0]):
            cv2.line(mask, (int(start[0]), int(start[1])),
                     (int(end[0]), int(end[1])), 255,
                     thickness=THICKNESS, lineType=cv2.LINE_8)
        else:
            logging.warning(f"Skipping out-of-bounds segment: {segment}")

    return np.nonzero(mask) if np.any(mask) else None


def get_masked(line_masks, border_masks, building_masks, mask_shape):
    """Combine masks and visualize if needed."""
    mask_img = np.zeros(mask_shape, dtype=np.uint8)  # Use the mask shape from JSON
    if line_masks is not None:
        mask_img[line_masks[0], line_masks[1]] = 255
    if building_masks is not None:
        mask_img[building_masks[0], building_masks[1]] = 255
    if border_masks is not None:
        mask_img[border_masks[0], border_masks[1]] = 255

    return mask_img

def get_attachment_name(path):
    """Extract attachment name from a path."""
    return path.split('/')[2]




