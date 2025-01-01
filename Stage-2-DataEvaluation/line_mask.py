import json
import logging
import zipfile
import networkx as nx
from config import (IN_DIR, s3_client, S3_BUCKET_NAME, S3_MAIN_DIR, OUT_DIR, lines_dir_1,
                    lines_dir_2, borders_dir_1, borders_dir_2, buildings_dir_1, buildings_dir_2, save_mask_to_s3,
                    generate_line_mask, S3_LOG_DIR)
from io import BytesIO
from tempfile import gettempdir
import os

# Flags to control mask generation
GENERATE_LINE_MASKS = True
GENERATE_BORDER_MASKS = True
GENERATE_BUILDING_MASKS = True

THICKNESS = 8

# Get a cross-platform temporary directory and define lof file path
temp_dir = gettempdir()  # This dynamically retrieves the temp directory (e.g., /tmp on Linux, C:\Temp on Windows)
log_file_path = os.path.join(temp_dir, "line_border_build_mask_generator.log")

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


def generate_lines_from_json(obs, attachment):
    """Extract line segments from JSON and generate a line mask."""
    lines = {k: v for k, v in obs.get('lines', {}).items()
             if v.get('attachment') == attachment}
    points_dict = obs['points']
    dimensions = obs['attachments'][attachment]['properties']['dimensions']  # Extract dimensions
    # Swap dimensions if necessary (x * y -> y * x)
    image_shape = (dimensions[1], dimensions[0])  # Use (y, x) for mask creation
    # Use the dimensions for mask creation

    segments = [
        [points_dict[start]['position'], points_dict[stop]['position']]
        for line, details in lines.items()
        for start, stop in zip(details['points'], details['points'][1:])
    ]

    logging.info(f"Extracted {len(segments)} line segments for {attachment}.")
    return generate_line_mask(segments, image_shape)


def generate_borders_from_json(obs, attachment):
    """Extract border segments from JSON and generate a border mask."""
    semantic_lines = {k: v for k, v in obs.get('semantic_lines', {}).items()
                      if v.get('attachment') == attachment}
    points_dict = obs['points']
    dimensions = obs['attachments'][attachment]['properties']['dimensions']  # Extract dimensions
    # Swap dimensions if necessary (x * y -> y * x)
    image_shape = (dimensions[1], dimensions[0])  # Use (y, x) for mask creation
    # Use the dimensions for mask creation

    segments = [
        [points_dict[start]['position'], points_dict[stop]['position']]
        for line, details in semantic_lines.items()
        for start, stop in zip(details['points'], details['points'][1:])
        if start in points_dict and stop in points_dict
    ]

    logging.info(f"Extracted {len(segments)} border segments for {attachment}.")
    return generate_line_mask(segments, image_shape)


def generate_building_from_json(obs, attachment):
    """Generate building masks from JSON using graph-based shortest paths."""
    points_dict = obs['points']
    buildings = {k: v for k, v in obs.get('buildings', {}).items()
                 if v.get('attachment') == attachment}
    dimensions = obs['attachments'][attachment]['properties']['dimensions']  # Extract dimensions
    # Swap dimensions if necessary (x * y -> y * x)
    image_shape = (dimensions[1], dimensions[0])  # Use (y, x) for mask creation
    # Use the dimensions for mask creation

    g = nx.Graph()
    g.add_nodes_from(points_dict.keys())

    for line in obs.get('lines', {}).values():
        line_points = line['points']
        for i in range(1, len(line_points)):
            g.add_edge(line_points[i - 1], line_points[i])

    segments = []
    for building in buildings.values():
        for i in range(1, len(building['points'])):
            try:
                path = nx.shortest_path(
                    g, building['points'][i - 1], building['points'][i]
                )
                segments.extend([[path[k - 1], path[k]] for k in range(1, len(path))])
            except nx.NetworkXNoPath:
                logging.warning("No path found between points.")
                segments.append([building['points'][i - 1], building['points'][i]])

    segments = [[points_dict[s[0]]['position'], points_dict[s[1]]['position']]
                for s in segments]
    logging.info(f"Extracted {len(segments)} building segments for {attachment}.")
    return generate_line_mask(segments, image_shape)


def read_zip(s3_bucket, s3_key, output_bucket):
    """Read a zip file from S3 and process its contents."""
    logging.info(f"fetching ZIP file from S3: s3://{s3_bucket}/{s3_key}")

    zip_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
    zip_data_stream = zip_obj['Body']  # This is a streaming body (not the entire file in memory)
    with zipfile.ZipFile(BytesIO(zip_data_stream.read()), 'r') as archive:
        prefix_1 = 'observations/snapshots/latest/'
        postfix_1 = '.json'
        prefix_2 = 'observations/snapshots/LineDetector/'
        postfix_2 = '.json'
        prefix_3 = 'observations/snapshots/BuildingDetection/'  # New prefix
        postfix_3 = '.json'  # New postfix

        # Process the first prefix and postfix for all masks (line, border, and building)
        process_files(archive, prefix_1, postfix_1, output_bucket, lines_dir_1, borders_dir_1, buildings_dir_1)
        # Process the second prefix and postfix for line masks only
        process_files(archive, prefix_2, postfix_2, output_bucket, lines_dir_2, borders_dir_2, buildings_dir_2,
                      line_only=True)
        # Process the third prefix and postfix for border and building masks
        process_files(archive, prefix_3, postfix_3, output_bucket, lines_dir_2, borders_dir_2, buildings_dir_2,
                      border_and_building=True)


def process_files(archive, prefix, postfix, output_bucket, lines_dir, borders_dir, buildings_dir, line_only=False,
                  border_and_building=False):
    """Process files within a zip archive based on specified prefixes and postfixes."""
    sketch_files = [x for x in archive.namelist() if x.startswith(prefix) and x.endswith(postfix)]
    if sketch_files:
        logging.info(f"Processing files with prefix: {prefix} and postfix: {postfix}")

    for i, sketch_file in enumerate(sketch_files):
        sketch_name = sketch_file[len(prefix):-len(postfix)]
        logging.info(f'Processing sketch: {i}: {sketch_name}.')

        # Process the files
        with archive.open(sketch_file, 'r') as afh:
            json_data = json.loads(afh.read())

        print(f"Processing sketch: {sketch_name}")

        for attachment in json_data['attachments']:
            try:
                # Check and extract dimensions
                dimensions = json_data['attachments'][attachment]['properties']['dimensions']
                height, width = dimensions[1], dimensions[0]
                image_shape = (height, width)  # Use (y, x) for mask creation
            except KeyError:
                logging.warning(f"Missing 'dimensions' key for attachment: {attachment}. Skipping...")
                continue

            # Generate masks based on flags passed
            line_masks = None
            border_masks = None
            building_masks = None

            if GENERATE_LINE_MASKS:
                line_masks = generate_lines_from_json(json_data, attachment) if not line_only else None
            if GENERATE_BORDER_MASKS and (not line_only or border_and_building):
                border_masks = generate_borders_from_json(json_data, attachment)
            if GENERATE_BUILDING_MASKS and (not line_only or border_and_building):
                building_masks = generate_building_from_json(json_data, attachment)

            output_bucket = f"{S3_BUCKET_NAME}"

            # Save masks to S3
            save_mask_to_s3(output_bucket, lines_dir, f"{sketch_name}_{attachment}.png", line_masks, image_shape)
            save_mask_to_s3(output_bucket, borders_dir, f"{sketch_name}_{attachment}.png", border_masks, image_shape)
            save_mask_to_s3(output_bucket, buildings_dir, f"{sketch_name}_{attachment}.png", building_masks,
                            image_shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        # List ZIP files in the S3 bucket
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=f"{S3_MAIN_DIR}/{IN_DIR}")
        for obj in response.get('Contents', []):
            s3_key = obj['Key']
            if s3_key.endswith('.zip'):  # Process only ZIP files
                logging.info(f"Processing ZIP files from S3: {s3_key}")
                read_zip(S3_BUCKET_NAME, s3_key, OUT_DIR)
    except Exception as e:
        logging.error(f"Error listing or processing ZIP files from S3: {e}")

    # Upload the log file to S3 after processing is done
    try:
        s3_log_key = f"{S3_MAIN_DIR}/{S3_LOG_DIR}/line_border_build_mask_generator.txt"
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





