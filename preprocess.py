import os
import numpy as np
import pandas as pd
import torch
import time
import torch.utils.data
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pickle
import skimage.io

MAP_ID_TILES = {}

class CancerDataset(torch.utils.data.Dataset):
    """This dataset includes the data for a list of subjects"""
    def __init__(self, data_path, mode, num_of_tiles, size_of_tile):
        """
        Args:
            data_path: (str) path to the images directory.
            mode: (str) can be train/test.
            transform: Optional, transformations applied to the tensor
        """
        self.data_path = data_path
        self.mode = mode
        self.num_of_tiles = num_of_tiles
        self.size_of_tile = size_of_tile
        self.img_dir = os.path.join(self.data_path, self.mode, self.mode)
        if mode =='train':
            self.mask_dir = os.path.join(self.data_path, f"{self.mode}_label_masks", f"{self.mode}_label_masks",)

        self.data_df = pd.read_csv(os.path.join(self.data_path, f"{self.mode}.csv"), sep=',')


    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        """
        Args:
            idx: (int) the index of the subject/session whom data is loaded.
        Returns:
                target: (dict) corresponding to data described by the following keys:
                image: (Tensor) MR image
                image_path: (str) path of the image
                image_id: (str) id of the image
                data_provider: (str) data provider of the sample
                tiles: tiles of the image
                gleason_score: (int) the cancer severity, if split = train
                isup_grade: (int) cancer score, if split = train
                mask_path: (str) path of the masked image, if split = train
        """

        data_provider = self.data_df.loc[idx, 'data_provider']
        image_id = self.data_df.loc[idx, 'image_id']
        image_path = os.path.join(self.img_dir, f'{image_id}.tiff')

        image = skimage.io.MultiImage(os.path.join(self.img_dir, f'{image_id}.tiff'))[1]


        target = {'image': image,
                  'image_path' : image_path,
                  'image_id': image_id,
                  'data_provider' : data_provider}

        if self.mode == 'train':
            isup_grade = self.data_df.loc[idx, 'isup_grade']
            gleason_score = self.data_df.loc[idx, 'gleason_score']

            target['gleason_score'] = gleason_score
            target['isup_grade'] = isup_grade

            mask_path = os.path.join(self.mask_dir, f'{image_id}.tiff')
            target['mask_path'] = mask_path

            isExist = os.path.exists(mask_path)
            if isExist:
                mask = skimage.io.MultiImage(mask_path)[1]
                tiles = get_tiles(image, mask, self.num_of_tiles, self.size_of_tile)
                target['tiles'] = tiles
                MAP_ID_TILES[image_id]= tiles

        elif self.mode == 'test':
            tiles = get_tiles(image, None, self.num_of_tiles, self.size_of_tile)
            MAP_ID_TILES[image_id]= tiles

        return image, target

def dataset_process(data_path, mode, batch_size=1, shuffle=False, save_tiles=False, num_of_tiles=36, size_of_tile=256):

    data_loader = torch.utils.data.DataLoader(
        CancerDataset(
            data_path,
            mode,
            num_of_tiles,
            size_of_tile
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    if save_tiles:
        for idx, (img, target) in tqdm(enumerate(data_loader)):
            pass

    save_map_id_tiles(mode)

    return data_loader



def get_tiles(image, mask, num_of_tiles, size_of_tile):
    tiles = []
    height, width, depth  = image.shape
    # how much should we pad the height and width so that the dimensions are multiple of
    pad_height = (size_of_tile - height % size_of_tile) % size_of_tile
    pad_width = (size_of_tile - width % size_of_tile) % size_of_tile
    # pad the image
    image = np.pad(image,[[pad_height // 2, pad_height - pad_height // 2], [pad_width // 2, pad_width - pad_width // 2],[0,0]],
                constant_values=255)
    #pad the mask
    if mask is not None:
        mask = np.pad(mask,[[pad_height // 2, pad_height - pad_height // 2],[pad_width // 2, pad_width - pad_width // 2],[0,0]],
                constant_values=0)

    image = image.reshape(image.shape[0] // size_of_tile,size_of_tile,image.shape[1] // size_of_tile,size_of_tile, 3)
    image = image.transpose(0,2,1,3,4).reshape(-1,size_of_tile,size_of_tile,3)

    if mask is not None:
        mask = mask.reshape(mask.shape[0] // size_of_tile, size_of_tile, mask.shape[1] // size_of_tile,size_of_tile, 3)
        mask = mask.transpose(0,2,1,3,4).reshape(-1,size_of_tile,size_of_tile,3)

    if len(image) < num_of_tiles:
        if mask is not None: mask = np.pad(mask,[[0,num_of_tiles-len(image)], [0,0],[ 0,0], [0,0]], constant_values=0)
        image = np.pad(image,[[0,num_of_tiles-len(image)], [0,0], [0,0], [0,0]], constant_values=255)

    idxs = np.argsort(image.reshape(image.shape[0],-1).sum(-1))[:num_of_tiles]
    image = image[idxs]

    if mask is not None: mask = mask[idxs]

    for i in range(len(image)):
        tiles.append({'idx':i, 'image':image[i], 'mask':mask[i]}) if mask is not None else tiles.append({'idx':i, 'image':image[i]})

    return tiles


def save_map_id_tiles(mode):
    a_file = open(f'map_id_tiles_{mode}.pkl', "wb")
    pickle.dump(MAP_ID_TILES, a_file)
    a_file.close()
