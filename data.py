from typing import Any, List, Tuple, Dict
import config_parser
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from PIL import Image, ImageFont, ImageDraw 
from torchvision import transforms
import histhoTransform

class DataHandler:

    preprocess = histhoTransform.Compose([
        histhoTransform.RandomChoice([
            histhoTransform.HEDJitter(theta=0.1), 
            histhoTransform.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5)),
            histhoTransform.HEDJitter(theta=0.0), 
            histhoTransform.RandomGaussBlur(radius=[0.5, 1.5]),
        ]),
        histhoTransform.ToTensor(),
        histhoTransform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    null_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    DATA_TRANSFORMS = {
        'in_distribution': preprocess,
        'out_distribution': null_transform,
    }

    class ImageDataset(Dataset):
        def __init__(self, dataframe: pd.DataFrame, transform = None):
            config = config_parser.load_config(config_path='config.json')
            self.dataframe = dataframe
            self.transform = transform
            font_path = config["font_path"]
            self.font = ImageFont.truetype(font_path, 150, encoding="unic") 

        def __len__(self) -> int:
            return len(self.dataframe)
            
        def __getitem__(self, idx: int) -> Tuple[Any, int]:
            image = Image.open(self.dataframe.loc[idx]['file_paths']).convert('RGB')
            label = self.dataframe.loc[idx]['class_label']
            
            # Draw a text on the image for studying the shortcuts
            if False:
                draw = ImageDraw.Draw(image)
                draw.text((50, 10),str(label),(0,0,255), font = self.font)
                
            if self.transform:
                image = self.transform(image)
                
            return image, label

    @staticmethod
    def read_dataframes(root: str, names: List[str], columns: List[str]) -> pd.DataFrame:
        concatenated = pd.DataFrame(columns = columns)
        for name in names:
            df = pd.read_csv(f"{root}{name}.csv")
            concatenated = pd.concat([concatenated, df], axis=0, ignore_index=True)
        return concatenated

    @classmethod
    def get_dataloaders(cls, holdout_trial_site: str, augmentation_in_training: bool, batch_size: int, num_workers: int = 64) -> Dict[str, DataLoader]:
        config = config_parser.load_config(config_path='config.json')
        dataframe_root = config['dataframe_root']
        TRIAL_SITES = config['trail_sites']
        df_columns = pd.read_csv(f'{dataframe_root}{holdout_trial_site}.csv').columns
        in_distribution_sites = [site for site in TRIAL_SITES if site != holdout_trial_site]
        
        transform = cls.DATA_TRANSFORMS['in_distribution'] if augmentation_in_training else cls.DATA_TRANSFORMS['out_distribution']

        in_distribution_df = cls.read_dataframes(dataframe_root, in_distribution_sites, df_columns)
        in_distribution_dataset = cls.ImageDataset(in_distribution_df, transform=transform)

        train_chunk, val_chunk = config['train_val_portions']
        train_portion = int(len(in_distribution_dataset) * train_chunk/100)
        val_portion = int(len(in_distribution_dataset) * val_chunk/100)
        test_portion = len(in_distribution_dataset) - val_portion - train_portion

        train_set, val_in_set, test_in_set = torch.utils.data.random_split(in_distribution_dataset, [train_portion, val_portion, test_portion])

        out_distribution_df = cls.read_dataframes(dataframe_root, [holdout_trial_site], df_columns)
        out_distribution_dataset = cls.ImageDataset(out_distribution_df, transform=cls.DATA_TRANSFORMS['out_distribution'])

        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            'val_in': DataLoader(val_in_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            'test_out': DataLoader(out_distribution_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        }

        return dataloaders
