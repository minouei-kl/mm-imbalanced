import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from io import BytesIO
from torchvision.transforms import transforms
from torchvision.utils import save_image

import tarfile


class RvlDataset(torch.utils.data.Dataset):
    def __init__(self, tar_path):
        super(RvlDataset, self).__init__()
        self.tar_path = tar_path

        self.imgs = []
        self.targets = []
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(size=(256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]
        )

        with tarfile.open(tar_path, mode="r|*") as stream:
            for info in stream:
                if not info.isfile():
                    continue
                file_path = info.name
                if file_path is None or 'png' not in file_path:
                    continue
                data = stream.extractfile(info).read()
                parent_dir_name = os.path.basename(
                    os.path.dirname(file_path))
                target = int(parent_dir_name)
                self.imgs.append(data)
                self.targets.append(target)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        with BytesIO(self.imgs[idx]) as stream:
            img = Image.open(stream).convert('RGB')
            img = self.image_transform(img)
        label = self.targets[idx]
        label = torch.tensor(label)
        return img, label


# train_dataset = RvlDataset(
#     tar_path="/home/minouei/Downloads/datasets/rvl/test.tar")
# labels = train_dataset.targets
# print(torch.bincount(torch.as_tensor(labels)).tolist())

# print(train_dataset[1510])
# im = train_dataset[1510][0][0]
# # im.show()
# save_image(im, 'img1.png')
