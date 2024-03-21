import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from PIL import Image
import pytorch_lightning as L


class SpacecraftDataset(torch.utils.data.Dataset):
    """
    Dataset module to import and pre-process the Spacecraft image data.
    """
    def __init__(self, labels_df, meta_df, imgs_dir=None, transforms=None, resize=None):
        self.imgs_dir = imgs_dir
        self.labels_df = labels_df # dataframe of indexes and bbox coordinates
        self.meta_df = meta_df
        self.img_idxs = labels_df.index.tolist()
        self.transforms = transforms
        self.resize = resize # resize the images for faster prototyping
    
    def __getitem__(self, idx):
        # Get image id and path
        img_id = self.img_idxs[idx]
        img_path = str(Path(self.imgs_dir, img_id + '.png'))
        n_objs = 1
        
        # Load image
        img = Image.open(img_path)

        # Get spacecraft id
        labels = torch.ones(n_objs, dtype=torch.int64)

        # Get bbox coordinates
        bbox = self._get_bbox(img_id)

        # Resize image and bounding box if resize parameter is provided
        if self.resize is not None:
            img, bbox = self._resize(img, bbox)
        
        # Convert data to format needed by model.
        target = {}
        target['boxes'] = torch.from_numpy(bbox).reshape((1,4))
        target['labels'] = labels
        target['image_id'] = img_id

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target
    
    def __len__(self):
        return len(self.img_idxs)

    def _get_bbox(self, image_id):
        return self.labels_df.loc[image_id].loc[["xmin", "ymin", "xmax", "ymax"]].values.astype('int')
    
    def _resize(self, img, bbox):
        # Resize image
        img = img.resize((int(img.size[0]*self.resize), int(img.size[1]*self.resize)), resample=Image.Resampling.LANCZOS)

        # Resize bounding box
        bbox[0] = int(bbox[0] * self.resize)
        bbox[1] = int(bbox[1] * self.resize)
        bbox[2] = int(bbox[2] * self.resize)
        bbox[3] = int(bbox[3] * self.resize)

        return img, bbox
    
# Define collate function to manage dictionary style dataset.
def collate_fn(batch):
    return list(zip(*batch))

class SpacecraftDataModule(L.LightningDataModule):
    """
    Lightning data module to pass Spacecraft data to Lightning models.
    """
    def __init__(self, dataset=None, batch_size=4):
        super().__init__()
        self.dataset = dataset
        self.batch_size=batch_size
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_ds = Subset(self.dataset, torch.arange(21000))
        self.valid_ds = Subset(self.dataset, torch.arange(21000, 25801))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )