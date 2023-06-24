import os
import json

def filter_json_by_image_names(json_file, image_folder, output_file):
    image_names = [os.path.splitext(filename)[0] for filename in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, filename))]
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    filtered_data = {"annotations": []}
    for annotation in data["annotations"]:
        image_id = os.path.splitext(annotation["image_id"])[0]
        if image_id in image_names:
            filtered_data["annotations"].append(annotation)
    
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f)

# Example usage
json_file = 'RAW-NOD-main/annotations/Sony/Raw_json_train.json'  # Path to the input JSON file
image_folder = "Train_raw"  # Path to the folder containing images
output_file = "filtered.json"  # Path to the output filtered JSON file


# json_path = 'RAW-NOD-main/annotations/Sony/Raw_json_train.json'
# output_folder = "Test_preprocessed"
# input_folder = "Test_raw"

filter_json_by_image_names(json_file, image_folder, output_file)
