import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from io import BytesIO
from torchvision.transforms import transforms
from torchvision.utils import save_image
from transformers import AutoTokenizer
import json

import tarfile

# 'roberta-base'  # 'bert-base-cased'
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 512


class RvlDataset(torch.utils.data.Dataset):
    def __init__(self, tar_path):
        super(RvlDataset, self).__init__()
        self.tar_path = tar_path

        self.imgs = []
        self.txts = []
        self.targets = []
        self.tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(size=(300, 300)),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [
                #                      0.229, 0.224, 0.225]),
            ]
        )
        with open(tar_path.replace('tar', 'json'), "rb") as json_file:
            json_data = json.load(json_file)

        with tarfile.open(tar_path, mode="r|*") as stream:
            for info in stream:
                if not info.isfile():
                    continue
                file_path = info.name
                if file_path is None or 'png' not in file_path:
                    continue
                filename = os.path.basename(file_path)
                if not filename in json_data.keys():
                    continue
                data = stream.extractfile(info).read()
                parent_dir_name = os.path.basename(
                    os.path.dirname(file_path))
                target = int(parent_dir_name)
                self.imgs.append(data)
                self.targets.append(target)
                txt = json_data[filename]
                txt = self.pre_processing_BERT(txt)
                self.txts.append(txt)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        txt = self.txts[idx]
        # self.pre_processing_BERT(txt)
        tensor_input_id, tensor_input_mask = txt
        with BytesIO(self.imgs[idx]) as stream:
            img = Image.open(stream).convert('RGB')
            img = self.image_transform(img)
        label = self.targets[idx]
        label = torch.tensor(label)
        # return img, label
        sample = {'image': img, 'BERT_ip': [
            tensor_input_id, tensor_input_mask], 'label': label}
        return sample

    def pre_processing_BERT(self, sent):
        # Create empty lists to store outputs
        input_ids = []
        attention_mask = []

        encoded_sent = self.tokenizer.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            pad_to_max_length=True,        # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,      # Return attention mask
            truncation=True
        )

        input_ids = encoded_sent.get('input_ids')
        attention_mask = encoded_sent.get('attention_mask')

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)

        return input_ids, attention_mask


# train_dataset = RvlDataset(
#     tar_path="/home/minouei/Downloads/datasets/rvl/test.tar")
# labels = train_dataset.targets
# print(torch.bincount(torch.as_tensor(labels)).tolist())

# print(train_dataset[5250])
# im = train_dataset[1510][0][0]
# # im.show()
# save_image(im, 'img1.png')
