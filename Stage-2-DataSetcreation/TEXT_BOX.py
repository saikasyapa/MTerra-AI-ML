import os
import cv2
import json
import zipfile
import logging
import datetime
import matplotlib.pyplot as plt
import boto3
import numpy as np
import pycococreatortools
from io import BytesIO

S3_BUCKET_NAME = "kadaster-magnasoft"
S3_MAIN_DIR = "Kadaster-AI-ML"
IN_DIR = "vector-data"
OUT_DIR = r'retrain_data/textbox/label'

s3_client = boto3.client('s3')

TRAIN_SPLIT = 70
TEST_SPLIT = 15
VALIDATION_SPLIT = 15

TARGET_SHAPE = (1664, 1024)
# IN_DIR = r'D:\DATA\Json Files'

os.makedirs(OUT_DIR, exist_ok=True)

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


def generate_box_mask(box, mask_shape):
    rct = ((box[0][0], box[0][1]), (box[1][0], box[1][1]), box[2])
    corner_points = cv2.boxPoints(rct).astype(np.int32)
    mask = np.zeros(mask_shape, dtype=np.uint8)
    mask = cv2.fillConvexPoly(mask, corner_points, 1)

    mask = cv2.resize(mask, (TARGET_SHAPE[1], TARGET_SHAPE[0])).astype(np.uint8)

    return mask


def generate_parcel_mask_from_json(obs, color, image_shape):
    filtered_texts = {k: item for k, item in obs['text'].items() if item['type'] == 'parcel' and item['color'] == color}
    masks = [generate_box_mask(filtered_texts[text]['box'], image_shape) for text in filtered_texts]

    return masks


def generate_mask_from_json(obs, text_type, image_shape):
    filtered_texts = {key: item for key, item in obs['text'].items() if item['type'] == text_type}
    masks = [generate_box_mask(filtered_texts[text]['box'], image_shape) for text in filtered_texts]

    return masks


def visualize_masks(img, measurement_masks, red_parcel_number_masks, black_parcel_number_masks,
                    blue_parcel_number_masks, coordinate_masks, year_masks):
    alpha = 0.4
    mask_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if measurement_masks is not None:
        for mask in measurement_masks:
            mask_img[mask] = mask_img[mask] * [2., 1., 1.] * alpha + (1 - alpha)

    if red_parcel_number_masks is not None:
        for mask in red_parcel_number_masks:
            mask_img[mask] = mask_img[mask] * [1., 2., 1.] * alpha + (1 - alpha)

    if black_parcel_number_masks is not None:
        for mask in black_parcel_number_masks:
            mask_img[mask] = mask_img[mask] * [1., 1., 2.] * alpha + (1 - alpha)

    if blue_parcel_number_masks is not None:
        for mask in blue_parcel_number_masks:
            mask_img[mask] = mask_img[mask] * [2., 2., 1.] * alpha + (1 - alpha)

    if coordinate_masks is not None:
        for mask in coordinate_masks:
            mask_img[mask] = mask_img[mask] * [2., 1., 2.] * alpha + (1 - alpha)

    if year_masks is not None:
        for mask in year_masks:
            mask_img[mask] = mask_img[mask] * [1., 2., 2.] * alpha + (1 - alpha)

    plt.figure(figsize=(4, 4))
    plt.imshow(mask_img)
    plt.show()


def get_parcel_numbers(json_data):
    parcel_numbers = set()
    for text_id, data in json_data['text'].items():
        if data['type'] == 'parcel' and 'value' in data:
            parcel_numbers.add(data['value'])

    return parcel_numbers


INFO = {
    "year": 2024,
    "version": "0.1.0",
    "description": "KKN Dataset",
    "contributor": "Wim Florijn",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

categories = {}

coco_output = {
    "info": INFO,
    "images": [],
    "annotations": [],
    "licenses": LICENSES,
}

image_id = 0
annotation_id = 0


def read_zip(s3_bucket,s3_key, measurement_masks=True, parcel_number_masks=True, coordinate_masks=True,
             year_masks=True):
    global image_id, annotation_id

    zip_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
    with zipfile.ZipFile(BytesIO(zip_obj['Body'].read()), 'r') as archive:
        prefix, postfix = 'observations/snapshots/latest/', '.latest.json'
        sketch_files = list(filter(lambda x: x.startswith(prefix) and x.endswith(postfix), archive.namelist()))
        for i, sketch_file in enumerate(sketch_files):
            sketch_name = sketch_file[len(prefix):-len(postfix)]
            logging.info(f'Processing sketch: {i}: {sketch_name}.')

            with archive.open(sketch_file, 'r') as fh:
                json_data = json.loads(fh.read())

            # Directly attempt to extract image shape from JSON
            # Initialize a list to store processed attachments
            processed_attachments = []

            for attachment, details in json_data['attachments'].items():
                try:
                    # Check if 'vectorize' is present and True in the attachment's details
                    if details.get('properties', {}).get('vectorize',
                                                         False):  # Check if vectorize is True in properties
                        properties = details['properties']

                        # Safely access 'dimensions' if it exists
                        dimensions = properties.get('dimensions', None)
                        if dimensions:
                            height, width = dimensions[1], dimensions[
                                0]  # Assuming dimensions[1] = height and dimensions[0] = width
                            image_shape = (height, width)

                            # Collect the processed attachment
                            processed_attachments.append({
                                "attachment": attachment,
                                "image_shape": image_shape
                            })
                            logging.info(f"Processed {attachment}: {image_shape}")
                        else:
                            logging.warning(f"Missing 'dimensions' for attachment {attachment}. Skipping...")
                    else:
                        logging.info(f"Skipping attachment {attachment} as vectorize is False.")
                except KeyError as e:
                    logging.warning(f"Missing key {e} for attachment {attachment}. Skipping...")
                except Exception as e:
                    logging.error(f"Unexpected error processing attachment {attachment}: {e}")
                continue
            # Log the collected processed attachments
            logging.info(f"Processed Attachments: {processed_attachments}")

            categories_to_instances = {}
            if parcel_number_masks:
                categories_to_instances['red_parcel'] = \
                    generate_parcel_mask_from_json(json_data, 'red', image_shape)

            if parcel_number_masks:
                categories_to_instances['blue_parcel'] = \
                    generate_parcel_mask_from_json(json_data, 'blue', image_shape)

            if parcel_number_masks:
                categories_to_instances['black_parcel'] = \
                    generate_parcel_mask_from_json(json_data, 'black', image_shape)

            if measurement_masks:
                categories_to_instances['measurement'] = \
                    generate_mask_from_json(json_data, 'measurement', image_shape)

            if coordinate_masks:
                categories_to_instances['coordinate'] = \
                    generate_mask_from_json(json_data, 'coordinate', image_shape)

            if year_masks:
                categories_to_instances['year'] = \
                    generate_mask_from_json(json_data, 'year', image_shape)

            for processed_attachment in processed_attachments:
                # Extract the attachment name for each processed attachment
                attachment = processed_attachment['attachment']
                # Construct the file name for the processed attachment
                file_name = f'{sketch_name}.{attachment}.jpg'
                # Create the COCO image info using the fixed target shape
                image_info = pycococreatortools.create_image_info(
                    image_id, file_name, TARGET_SHAPE  # Use the predefined TARGET_SHAPE
                )

                # Append the created image info to the COCO output
                coco_output["images"].append(image_info)

            for category, masks in categories_to_instances.items():
                if category not in categories:
                    categories[category] = len(categories) + 1

                class_id = categories[category]
                for index, binary_mask in enumerate(masks):
                    category_info = {'id': class_id, 'is_crowd': False}
                    annotation_info = pycococreatortools.create_annotation_info(
                        annotation_id, image_id, category_info, binary_mask, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    annotation_id += 1
            image_id += 1



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s;%(levelname)s;%(message)s')

    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix =f"{S3_MAIN_DIR}/{IN_DIR}")
    for obj in response.get('Contents', []):
            s3_key = obj['Key']
            if s3_key.endswith('.zip'): # Process only ZIP files
                logging.info(f"Processing ZIP files from S3: {s3_key}")
                read_zip(S3_BUCKET_NAME, s3_key)

    coco_output['categories'] = [{
        'id': category_id,
        'name': category_name,
        'supercategory': 'sketch',
    } for category_name, category_id in categories.items()]

    images_length = len(coco_output['images'])
    train, validate, test = np.split(np.asarray(coco_output['images']), [
        int(images_length * (TRAIN_SPLIT / 100)), int(images_length * ((TRAIN_SPLIT / 100) + (TEST_SPLIT / 100)))
    ])

    # Prepare output file paths in S3
    train_key = f"{S3_MAIN_DIR}/{OUT_DIR}/train_annotations.json"
    validate_key = f"{S3_MAIN_DIR}/{OUT_DIR}/validate_annotations.json"
    test_key = f"{S3_MAIN_DIR}/{OUT_DIR}/test_annotations.json"

    # Save JSON data directly to S3
    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=train_key,
        Body=json.dumps({**coco_output, 'images': train.tolist()})
    )
    logging.info(f"Train annotations saved to S3: {train_key}")

    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=validate_key,
        Body=json.dumps({**coco_output, 'images': validate.tolist()})
    )
    logging.info(f"Validate annotations saved to S3: {validate_key}")

    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=test_key,
        Body=json.dumps({**coco_output, 'images': test.tolist()})
    )
    logging.info(f"Test annotations saved to S3: {test_key}")


