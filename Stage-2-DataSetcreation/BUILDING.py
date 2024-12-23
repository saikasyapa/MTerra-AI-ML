# building_mask_generator.py

import logging
import networkx as nx
from pathlib import Path
from common import read_zip, generate_line_mask, get_masked, TARGET_SHAPE, IN_DIR, S3_BUCKET_NAME, S3_MAIN_DIR, upload_image_to_s3, s3_client

# OUT_DIR = r'D:\DATA\RETRAINING\building_masks'
# Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
# S3 Configuration
S3_SUB_DIR = "retrain_data/building/label"


def generate_building_from_json(obs, sketch_name, image_shape, attachment):
    points_dict, buildings = obs['points'], obs['buildings']
    buildings = {k: v for k, v in buildings.items() if v.get('attachment') == attachment}
    g = nx.Graph()
    g.add_nodes_from(obs['points'].keys())
    for line in obs['lines'].values():
        line_points = line['points']
        for i in range(1, len(line_points)):
            g.add_edge(line_points[i - 1], line_points[i])
    segments = []
    for building in buildings.values():
        building = building['points']
        for i in range(1, len(building)):
            try:
                path = nx.shortest_path(g, building[i - 1], building[i])
                for k in range(1, len(path)):
                    segments.append([path[k - 1], path[k]])
            except nx.exception.NetworkXNoPath:
                segments.append([building[i - 1], building[i]])
    segments = [[points_dict[segment[0]]['position'], points_dict[segment[1]]['position']] for segment in segments]
    building_masks = generate_line_mask(segments, image_shape)
    if building_masks:
        masked_img = get_masked(None, building_masks, None, TARGET_SHAPE)
        s3_key = f"{S3_MAIN_DIR}/{S3_SUB_DIR}/{sketch_name}.{attachment}.png"
        
        # Upload to S3
        upload_image_to_s3(masked_img, S3_BUCKET_NAME, s3_key)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s;%(levelname)s;%(message)s')

    try:
        # listing ZIP files in the S3 Bucket
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=f"{S3_MAIN_DIR}/{IN_DIR}")
        for obj in response.get('Contents', []):
            s3_key = obj['Key']
            if s3_key.endswith('.zip'):  # Process only ZIP files
                logging.info(f"process ZIP file in S3: {s3_key}")
                read_zip(S3_BUCKET_NAME,s3_key,generate_building_from_json)
    except Exception as e:
        logging.error(f"Error listing or processing ZIP files: {e}")

