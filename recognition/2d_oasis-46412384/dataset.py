import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from os import listdir
from PIL import Image


class DataError(Exception):
    """
    Raised when a problem regarding the dataset is encountered
    """
    pass


class OasisDataset(Dataset):
    """
    Dataset for OASIS greyscale images and segmentations
    """

    def __init__(self, data_dir, mask_dir, subset_size=-1):
        self.data_paths = [data_dir + "/" + file for file in listdir(data_dir)[:subset_size]]
        self.mask_paths = [mask_dir + "/" + file for file in listdir(mask_dir)[:subset_size]]

        if (subset_size < 0 or len(self.data_paths) < subset_size):
            self.subset_size = len(self.data_paths)
        else:
            self.subset_size = subset_size
        
        if (len(self.mask_paths) != self.subset_size):
            raise DataError("Non-matching subset sizes between data (length %d) and mask (length %d)." % (len(self.data_paths), len(self.mask_paths)))

    def __len__(self):
        return self.subset_size

    def __getitem__(self, idx):
        if (idx < 0 or idx >= self.subset_size):
            raise IndexError("Index %d out of range for subset size %d" % (idx, self.subset_size))

        # Read and load image data
        img = Image.open(self.data_paths[idx])
        img_data = np.resize(np.array(img.getdata(), dtype=np.float32), img.size)

        img_data = (img_data - img_data.mean()) / (img_data.std() + 1e-6)
        img_data = torch.from_numpy(img_data).unsqueeze(0).float()

        msk = Image.open(self.mask_paths[idx])
        msk_data = torch.from_numpy(OasisDataset.decode_mask(np.resize(np.array(msk.getdata(), dtype=np.int64), msk.size)))

        return img_data, msk_data


    """
    Decode greyscale mask values from numpy array to segment classification integers
    """
    def decode_mask(enc_msk):
        vals = set()
        for val in enc_msk.flat:
            vals.add(val)

        vals = np.sort(np.array(list(vals), dtype=np.uint8))
        
        dec_msk = np.zeros_like(enc_msk)
        for idx, val in enumerate(vals):
            dec_msk[enc_msk == val] = idx

        return dec_msk


    """
    Encode segment classification integers from numpy array into greyscale values of equal distribution
    """
    def encode_mask(dec_msk):
        vals = set()
        for val in dec_msk.flat:
            vals.add(val)
        
        vals = np.sort(np.array(list(vals)))

        enc_msk = np.array(dec_msk) * 255 / (len(vals) - 1)

        return enc_msk


def get_oasis_dataloaders(data_dir, batch_size, subset_size=-1):
    IMG_TRAIN_PATH = "keras_png_slices_train"
    MSK_TRAIN_PATH = "keras_png_slices_seg_train"
    IMG_VAL_PATH = "keras_png_slices_validate"
    MSK_VAL_PATH = "keras_png_slices_seg_validate"
    IMG_TEST_PATH = "keras_png_slices_test"
    MSK_TEST_PATH = "keras_png_slices_seg_test"

    ds_train = OasisDataset(data_dir + IMG_TRAIN_PATH, data_dir + MSK_TRAIN_PATH, subset_size=subset_size)
    ds_validate = OasisDataset(data_dir + IMG_VAL_PATH, data_dir + MSK_VAL_PATH, subset_size=subset_size // 2 if subset_size else -1)
    ds_test = OasisDataset(data_dir + IMG_TEST_PATH, data_dir + MSK_TEST_PATH, subset_size=subset_size // 2 if subset_size else -1)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_validate = DataLoader(ds_validate, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    return dl_train, dl_validate, dl_test