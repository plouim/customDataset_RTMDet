import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from sklearn.model_selection import train_test_split
import shutil

def xml_to_dict(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotation = {}
    annotation['filename'] = root.find('filename').text
    annotation['size'] = {
        'width': int(root.find('size/width').text),
        'height': int(root.find('size/height').text),
        'depth': int(root.find('size/depth').text),
    }
    annotation['objects'] = []

    for obj in root.findall('object'):
        obj_dict = {
            'name': obj.find('name').text,
            'bbox': [
                int(obj.find('bndbox/xmin').text),
                int(obj.find('bndbox/ymin').text),
                int(obj.find('bndbox/xmax').text),
                int(obj.find('bndbox/ymax').text),
            ],
        }
        annotation['objects'].append(obj_dict)
    return annotation

def custom_to_coco(annotations, images_dir, categories):
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': [{'id': i + 1, 'name': c} for i, c in enumerate(categories)],
    }

    img_id = 0
    ann_id = 0

    for ann_file in annotations:
        ann = xml_to_dict(ann_file)
        img_id += 1
        image = {
            'id': img_id,
            'file_name': ann['filename'],
            'width': ann['size']['width'],
            'height': ann['size']['height'],
        }
        coco_data['images'].append(image)

        for obj in ann['objects']:
            ann_id += 1
            annotation = {
                'id': ann_id,
                'image_id': img_id,
                'category_id': categories.index(obj['name']) + 1,
                'bbox': obj['bbox'],
                'area': (obj['bbox'][2] - obj['bbox'][0]) * (obj['bbox'][3] - obj['bbox'][1]),
                'iscrowd': 0,
            }
            coco_data['annotations'].append(annotation)
    return coco_data

def main():
    base_dir = './data/'
    annotations_dir = os.path.join(base_dir, 'Annotations')
    images_dir = os.path.join(base_dir, 'JPEGImages')
    output_dir = './COCO_format'

    annotation_files = [os.path.join(annotations_dir, f) for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    categories = [
        'Motor Vehicle',
        'Non-motorized Vehicle',
        'Pedestrian',
        'Traffic Light-Red Light',
        'Traffic Light-Yellow Light',
        'Traffic Light-Green Light',
        'Traffic Light-Off'
        ]

    train_files, testval_files = train_test_split(annotation_files, test_size=0.2, random_state=42)
    val_files, test_files = train_test_split(testval_files, test_size=0.5, random_state=42)

    for name, files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
        coco_data = custom_to_coco(files, images_dir, categories)
        split_output_dir = os.path.join(output_dir, name)
        split_images_dir = os.path.join(split_output_dir, 'images')
        os.makedirs(split_images_dir, exist_ok=True)
        
        for img_data in coco_data['images']:
            img_filename = img_data['file_name']
            shutil.copyfile(os.path.join(images_dir, img_filename), os.path.join(split_images_dir, img_filename))

        with open(os.path.join(split_output_dir, 'instances.json'), 'w') as f:
            json.dump(coco_data, f)

if __name__ == '__main__':
    main()

