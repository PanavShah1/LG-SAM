import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from transformers import BertModel, BertTokenizer

import config

from .backbone import MultiModalSwinTransformer
from .mask_predictor import SimpleDecoding


def transform_image(image: Image.Image) -> torch.Tensor:
    return transforms.Compose(
        [
            transforms.Resize((896, 896)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(image)  # type: ignore


DEFAULT_ARGS = {
    "bert_tokenizer": config.BERT_CHECKPOINT,
    "ck_bert": config.BERT_CHECKPOINT,
    "device": "cuda:0",
    "fusion_drop": 0.0,
    "img_size": 896,
    "mha": "",
    "model": "lavt_one",
    "model_id": "RemoteSAM",
    "swin_type": "base",
    "window12": True,
}


class LAVTOne(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(LAVTOne, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args["ck_bert"])
        self.text_encoder.pooler = None

    def forward(self, x, text, l_mask):
        input_shape = x.shape[-2:]
        ### language inference ###
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (6, 10, 768)
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l)
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        ##########################
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = (
            features  # e.g. x_c1:[B, 128, 120, 120], x_c2:[B, 256, 60, 60], x_c3:[B, 512, 30, 30], x_c4:[B, 1024, 15, 15]
        )
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=True)
        return x


def _segm_lavt_one(args):
    # initialize the SwinTransformer backbone with the specified version
    embed_dim = 128
    depths = [2, 2, 18, 2]
    num_heads = [4, 8, 16, 32]
    window_size = 12
    mha = [1, 1, 1, 1]

    out_indices = (0, 1, 2, 3)
    backbone = MultiModalSwinTransformer(
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=out_indices,
        use_checkpoint=False,
        num_heads_fusion=mha,
        fusion_drop=args["fusion_drop"],
        # frozen_stages=args.frozen_stages,
        # only_fusion=args.only_fusion,
    )

    classifier = SimpleDecoding(8 * embed_dim)
    model = LAVTOne(backbone, classifier, args)
    return model


def init_demo_model(checkpoint, device):
    args = DEFAULT_ARGS.copy()
    args["device"] = device
    args["window12"] = True

    model = _segm_lavt_one(args=args)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)["model"],
        strict=False,
    )
    model = model.to(device)
    return model


def embed_sentences(
    sentences: list[str], max_tokens: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    # init
    tokenizer = BertTokenizer.from_pretrained(config.BERT_CHECKPOINT)
    inputs = []
    attentions = []

    for sentence in sentences:
        sentence_tokenized = tokenizer.encode(text=sentence, add_special_tokens=True)
        sentence_tokenized = sentence_tokenized[:max_tokens]
        # pad the tokenized sentence
        padded_sent_toks = [0] * max_tokens
        padded_sent_toks[: len(sentence_tokenized)] = sentence_tokenized
        # create a sentence token mask: 1 for real words; 0 for padded tokens
        attention_mask = [0] * max_tokens
        attention_mask[: len(sentence_tokenized)] = [1] * len(sentence_tokenized)
        inputs.append(padded_sent_toks)
        attentions.append(attention_mask)

    return torch.tensor(inputs), torch.tensor(attentions)


class RemoteSAM:
    MLC_balance_factor = 0.5
    MCC_balance_factor = 1.0
    GMP = torch.nn.AdaptiveMaxPool2d(1)
    GAP = torch.nn.AdaptiveAvgPool2d(1)

    def __init__(
        self,
        RemoteSAM_model,
        device,
        *,
        MLC_balance_factor=0.5,
        MCC_balance_factor=1.0,
    ):
        self.RemoteSAM_model = RemoteSAM_model
        self.RemoteSAM_model.eval()
        self.device = device
        self.MLC_balance_factor = MLC_balance_factor
        self.MCC_balance_factor = MCC_balance_factor

    def referring_seg(
        self, original_image: Image.Image, text_prompt: str, return_prob=False
    ):
        original_image_size = original_image.size
        image = transform_image(original_image).unsqueeze(0).to(self.device)

        inputs, attentions = embed_sentences([text_prompt])
        inputs = inputs[0].to(self.device)
        attentions = attentions[0].to(self.device)

        output = self.RemoteSAM_model(image, inputs, l_mask=attentions)

        mask = output.cpu().argmax(1, keepdim=True)  # (1, 1, resized_shape)
        mask = torch.nn.functional.interpolate(
            mask.float(), original_image_size[::-1]
        )  # (1, 1, origin_shape)
        mask = mask.squeeze().data.numpy().astype(np.uint8)  # np(origin_shape)

        prob = torch.softmax(output, dim=1)[:, [1], :, :].cpu()  # (1, 1, resized_shape)
        prob = torch.nn.functional.interpolate(
            prob.float(), original_image_size[::-1]
        )  # (1, 1, origin_shape)
        prob = prob.squeeze().data.numpy()  # np(origin_shape)

        if return_prob:
            return mask, prob
        else:
            return mask

    def referring_seg_batch(
        self, images: list[Image.Image], text_prompts: list[str], return_prob=False
    ):
        """
        Process a batch of images and sentences for referring segmentation.

        Args:
            images: List of PIL Images
            text_prompts: List of text prompts (must match length of images)
            return_prob: Whether to return probability maps

        Returns:
            If return_prob=False: List of masks (numpy arrays)
            If return_prob=True: Tuple of (masks, probs) where each is a list
        """
        if len(images) != len(text_prompts):
            raise ValueError("Number of images and sentences must match")

        if len(images) == 0:
            return [] if not return_prob else ([], [])

        # Store original sizes for each image
        original_image_sizes = [img.size for img in images]

        # Transform and stack images
        image_batch = []
        for img in images:
            image_batch.append(transform_image(img))
        image_batch = torch.stack(image_batch).to(self.device)

        # Embed sentences
        inputs, attentions = embed_sentences(text_prompts)
        inputs = inputs.to(self.device)
        attentions = attentions.to(self.device)

        # Forward pass
        output = self.RemoteSAM_model(
            image_batch, inputs, l_mask=attentions
        )  # (B, 2, H, W)

        # Process each image in the batch
        masks = []
        probs = []

        for i in range(len(images)):
            original_size = original_image_sizes[i]

            # Extract mask for this image
            mask = (
                output[i : i + 1].cpu().argmax(1, keepdim=True)
            )  # (1, 1, resized_shape)
            mask = torch.nn.functional.interpolate(
                mask.float(), original_size[::-1]
            )  # (1, 1, origin_shape)
            mask = mask.squeeze().data.numpy().astype(np.uint8)  # np(origin_shape)
            masks.append(mask)

            if return_prob:
                prob = torch.softmax(output[i : i + 1], dim=1)[
                    :, [1], :, :
                ].cpu()  # (1, 1, resized_shape)
                prob = torch.nn.functional.interpolate(
                    prob.float(), original_size[::-1]
                )  # (1, 1, origin_shape)
                prob = prob.squeeze().data.numpy()  # np(origin_shape)
                probs.append(prob)

        if return_prob:
            return masks, probs
        else:
            return masks
