import os
import json
import zipfile
import logging
import boto3
from config import (S3_BUCKET_NAME, s3_client, OUT_DIR_TEXT_BOX,
                    OUT_DIR_TEXT_BOX1, S3_MAIN_DIR, OUT_DIR, IN_DIR, S3_LOG_DIR)
from io import BytesIO
from tempfile import gettempdir


# Get a cross-platform temporary directory and define lof file path
temp_dir = gettempdir()  # This dynamically retrieves the temp directory (e.g., /tmp on Linux, C:\Temp on Windows)
log_file_path = os.path.join(temp_dir, "text_box_detector_json.log")

# Ensure the directory for the log file exists
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
# Setting yup to a lof file
file_handler = logging.FileHandler(log_file_path)
stream_handler = logging.StreamHandler()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s;%(levelname)s;%(message)s',
    handlers=[file_handler, stream_handler]
)


def extract_text_data(json_data, sketch_name):
    """
    Extract relevant text data from the JSON.
    """
    extracted_data = []

    for text_id, text_info in json_data.get('text', {}).items():
        # If the type is 'parcel' and color is specified, update the type
        if text_info.get("type") == "parcel" and "color" in text_info:
            color = text_info.get("color")
            text_info["type"] = f"{color}_{text_info['type']}"  # Combine color and type

        extracted_data.append({
            "sketch_name": sketch_name,
            "box": text_info.get("box"),
            "type": text_info.get("type"),
            "color": text_info.get("color")  # Ensure color is included in the output
        })

    return extracted_data


def read_zip(zip_key, prefix, postfix, output_dir):
    logging.info(f'Reading zip file from s3 Bucket: {zip_key}')
    try:
        # Download zip file from s3
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=zip_key)
        with zipfile.ZipFile(BytesIO(response['Body'].read()), 'r') as archive:
            sketch_files = [
                x for x in archive.namelist()
                if x.startswith(prefix) and x.endswith(postfix)
            ]
            logging.info(f'Found {len(sketch_files)} sketch files with prefix "{prefix}" and postfix "{postfix}".')
            for i, sketch_file in enumerate(sketch_files):
                sketch_name = sketch_file[len(prefix):-len(postfix)]
                logging.info(f'Processing sketch: {i + 1}/{len(sketch_files)}: {sketch_name, postfix}.')

                try:
                    # Load JSON data
                    with archive.open(sketch_file, 'r') as fh:
                        json_data = json.loads(fh.read())

                    # Extract relevant text data
                    text_data = extract_text_data(json_data, sketch_name)

                    # Save extracted data of JSON files to S3
                    output_json_path = os.path.join(output_dir, f'{sketch_name}.json')
                    s3_client.put_object(
                        Bucket=S3_BUCKET_NAME,
                        Key=output_json_path,
                        Body=json.dumps(text_data, indent=4)
                    )
                    logging.info(f"Data is successfully saved in {S3_BUCKET_NAME} {output_json_path}")

                except Exception as e:
                    logging.error(f'Error processing sketch {sketch_name}: {e}')
    except Exception as e:
        logging.error(f'Error reading zip file {zip_key}: {e}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s;%(levelname)s;%(message)s')

    # List objects in the input bucket
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=f"{S3_MAIN_DIR}/{IN_DIR}")
        projects = response.get('Contents', [])
        if 'Contents' not in response:
            logging.error(f"No contents found in the {S3_MAIN_DIR}")
        else:
            for j, obj in enumerate(projects):
                zip_key = obj['Key']
                logging.info(f'Processing project {j + 1} / {len(projects)}: {zip_key}.')

                # Define prefixes and postfixes
                prefix_1 = 'observations/snapshots/latest/'
                postfix_1 = '.latest.json'
                prefix_2 = 'observations/snapshots/TextboxDetection/'
                postfix_2 = '.TextboxDetection.json'

                # Read zip for first prefix/postfix (ground truth)
                read_zip(zip_key, prefix_1, postfix_1, OUT_DIR_TEXT_BOX)

                # Read zip from for second prefix/postfix (predicted data)
                read_zip(zip_key, prefix_2, postfix_2, OUT_DIR_TEXT_BOX1)
    except Exception as e:
        logging.error(f"Error listing objects in bucket {IN_DIR}: {e}")

    # Upload the log file to S3 after processing is done
    try:
        s3_log_key = f"{S3_MAIN_DIR}/{S3_LOG_DIR}/text_box_detector_json.txt"

        # Upload log file to S3
        with open(log_file_path, 'rb') as log_file:
            s3_client.put_object(Body=log_file, Bucket=S3_BUCKET_NAME, Key=s3_log_key)
        logging.info(f"Log file uploaded successfully{S3_BUCKET_NAME} {s3_log_key}")

    except Exception as e:
        logging.error(f"Error uploading lof file to S3{S3_BUCKET_NAME} {s3_log_key}")

    finally:
        # Close the logging file handler
        file_handler.close()
        logging.getLogger().removeHandler(file_handler)
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
            logging.info("Cleaned up the local lof file.")


