import numpy as np
import os
import imageio
from PIL import Image
import json


def load_humbi_data(basedir, split='train', factor=1, read_offline=True):
    with open(os.path.join(basedir, f'transforms_{split}.json'), 'r') as fp:
        meta = json.load(fp)

    poses = []
    images = []
    image_paths = []

    for i, frame in enumerate(meta['frames']):
        img_path = os.path.abspath(os.path.join(basedir, frame['file_path']))
        poses.append(np.array(frame['transform_matrix']))
        image_paths.append(img_path)

        if read_offline:
            img = imageio.imread(img_path)
            H, W = img.shape[:2]
            if factor > 1:
                img = Image.fromarray(img).resize((W//factor, H//factor))
            images.append((np.array(img) / 255.).astype(np.float32))
        elif i == 0:
            img = imageio.imread(img_path)
            H, W = img.shape[:2]
            if factor > 1:
                img = Image.fromarray(img).resize((W//factor, H//factor))
            images.append((np.array(img) / 255.).astype(np.float32))

    poses = np.array(poses).astype(np.float32)
    images = np.array(images).astype(np.float32)

    H, W = images[0].shape[:2]
    # camera_angle_x = float(meta['camera_angle_x'])
    # focal = .5 * W / np.tan(.5 * camera_angle_x)
    focal_x = meta['focal_x']
    focal_y = meta['focal_y']

    return images, poses, [H, W, focal_x, focal_y], image_paths
