from transformers import AutoImageProcessor, ViTMAEModel
from PIL import Image

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)


class DETRdemo(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = torch.nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)

        print(h.shape)

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h),
                'pred_boxes': self.linear_bbox(h).sigmoid()}

class DETRMAE(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        self.num_classes = num_classes
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

        self.encoder = self.get_mae_encoder()
        self.decoder = self.get_detr_decoder()

        self.emb_linear = torch.nn.Linear(768, hidden_dim)


        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        self.query_pos = nn.Parameter(torch.rand(50, hidden_dim))


    def get_mae_encoder(self):
        return ViTMAEModel.from_pretrained("facebook/vit-mae-base")

    def get_detr_decoder(self):
        detr = DETRdemo(num_classes=self.num_classes)
        state_dict = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
            map_location='cpu', check_hash=True)
        detr.load_state_dict(state_dict)
        detr.eval()
        return detr.transformer.decoder

    def forward(self, inputs):
        
        # print(inputs.tensors)
        # print(inputs.mask)

        inputs = inputs.tensors
        # min = torch.min(inputs)
        # max = torch.max(inputs)
        # inputs = (inputs - min) / (max - min)
        # inputs.requires_grad=True
        # device = torch.device("cpu")
        # inputs = inputs.to(device)
        # print(inputs.device)

        # inputs = self.image_processor(images=inputs, return_tensors="pt", do_resize=False,)
        # inputs = self.image_processor(images=inputs, return_tensors="pt")
        # inputs.pixel_values = inputs.pixel_values.to(device)
        # inputs.head_mask = inputs.head_mask.to(device)
        # print(inputs.pixel_values.device)
        memory = self.encoder(pixel_values=inputs)
        last_hidden_states = memory.last_hidden_state
        # print(last_hidden_states.shape)
        last_hidden_states = self.emb_linear(last_hidden_states)
        tgt = self.query_pos.unsqueeze(0)
        h = self.decoder(tgt, last_hidden_states)
        # print(h.shape)

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h),
                'pred_boxes': self.linear_bbox(h).sigmoid()}