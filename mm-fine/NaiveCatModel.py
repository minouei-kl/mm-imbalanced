import torch
import torch.nn as nn
from transformers import AutoModel
from resnet import resnet50
import torch.nn.functional as F


class TextModel(nn.Module):

    def __init__(self, config):
        super(TextModel, self).__init__()
        bert_name = '/home/minouei/Documents/models/rvl/bert-im/bert_model4'
        # self.bert = AutoModel.from_pretrained(bert_name)
        self.bert = torch.load(bert_name)

        # 是否进行fine-tune
        # for param in self.bert.parameters():
        #     if config.fixed_text_model_params:
        #         param.requires_grad = False
        #     else:
        #         param.requires_grad = True
        for n, p in self.bert.named_parameters():
            if 'classifier' in n:
                pass
            else:
                p.requires_grad = False

    def forward(self, bert_inputs, masks, token_type_ids=None):
        assert bert_inputs.shape == masks.shape, 'error! bert_inputs and masks must have same shape!'
        bert_out = self.bert(
            input_ids=bert_inputs, token_type_ids=token_type_ids, attention_mask=masks)
        logits = bert_out['logits']

        return logits


class ImageModel(nn.Module):

    def __init__(self, config):
        super(ImageModel, self).__init__()
        resume = '/home/minouei/Documents/models/rvl/IB-Loss/ckpt/checkpoint/cifar10_resnet50_IBFocal_IBReweight_re_lr_adam/ckpt.pth.tar'
        self.full_resnet = resnet50(num_classes=16, channel=3)
        checkpoint = torch.load(resume, map_location='cuda:0')
        self.full_resnet.load_state_dict(checkpoint['state_dict'])

        # 是否进行fine-tune
        # for param in self.full_resnet.parameters():
        #     if config.fixed_image_model_params:
        #         param.requires_grad = False
        #     else:
        #         param.requires_grad = True
        for n, p in self.full_resnet.named_parameters():
            if 'fc' in n:
                pass
            else:
                p.requires_grad = False

    def forward(self, imgs):
        output, _ = self.full_resnet(imgs)
        return output


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # text
        self.text_model = TextModel(config)
        # image
        self.img_model = ImageModel(config)

        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.Linear(32, config.num_labels)
        )
        # self.loss_func = nn.CrossEntropyLoss()

    def forward(self, text, image, label=None):
        text_feature = self.text_model(text[0], text[1])
        img_feature = self.img_model(image)
        # text_feature = F.sigmoid(text_feature)
        # img_feature = F.sigmoid(img_feature)
        # print(text_feature)
        # print(img_feature)
        preds = self.classifier(
            torch.cat([text_feature, img_feature], dim=1)
        )

        return preds


class Model2(nn.Module):

    def __init__(self, config):
        super(Model2, self).__init__()
        self.classifier = nn.Sequential(

            nn.Linear(config.num_labels * 2, config.num_labels)
        )

    def forward(self, feature, label=None):
        preds = self.classifier(feature)

        return preds
