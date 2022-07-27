import numpy as np
import torch
import nibabel as nib
import os
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
import albumentations.augmentations.geometric.transforms as gt
import torch.optim.lr_scheduler

#custom data input
train_ag=A.Compose([
    A.Resize(height=250,width=250),
    A.RandomCrop(224,224,0.5),
    A.HorizontalFlip(p=0.5),
    #A.GaussNoise(var_limit=(10.0, 15.0),mean=-90,p=0.25),
    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=1.0),
    A.Blur(10, p=1.0),
    gt.ElasticTransform(alpha=1, sigma=50, alpha_affine=25, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, approximate=False, same_dxdy=False, p=0.5),
    A.Normalize(mean=(0),std=(1)),
])
val_ag=A.Compose([
    A.Resize(height=224,width=224),
    A.Normalize(mean=(0),std=(1))
])
class RV(Dataset):
    def __init__(self,image_dir,mask_dir,transform=None):
        self.image_dir=image_dir
        self.mask_dir=mask_dir
        self.transform = transform
        self.masks=os.listdir(mask_dir)

    def __len__(self):
        return len(self.masks)

    def __getitem__(self,index):
        mask_path=os.path.join(self.mask_dir,self.masks[index])
        img_path=os.path.join(self.image_dir,self.masks[index].replace("_gt.nii.gz",".nii.gz"))
        img=nib.load(img_path)
        image=np.array(img.dataobj)
        image=image.astype(np.float32)
        msk=nib.load(mask_path)
        mask=np.array(msk.dataobj)
        mask=mask.astype(np.float32)

        if self.transform is not None:
            augmentations=self.transform(image=image,mask=mask)
            image=augmentations["image"]
            mask=augmentations["mask"]
        
        image=torch.from_numpy(image).permute(2,0,1)
        mask=torch.from_numpy(mask).permute(2,0,1).squeeze(dim=0).to(dtype=torch.long)
        return image,mask