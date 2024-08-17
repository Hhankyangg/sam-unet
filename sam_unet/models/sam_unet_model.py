from typing import Any, Dict, List, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from sam_unet.models.wrapped import SAMImageEncodeWrapper, SAMPromptEncodeWrapper, SAMMaskDecoderWrapper_Med
from sam_unet.models.resnet import ResNet
from sam_unet.models.segment_anything.build_sam import sam_model_registry
from sam_unet.models.segment_anything.modeling.common import LayerNorm2d, Reshaper
from sam_unet.config import config_dict
from sam_unet.utils.utils import preprocess
    

class SAM_UNET(nn.Module):
    
    mask_threshold: float = 0.0
    
    def __init__(self,
                 resnet_type: int = 50,
                 ori_sam: nn.Module = sam_model_registry[f'vit_b_{config_dict["img_size"]}'](None),
                 is_resnet_pretrained_or_not: bool = False
                 ): 
        
        super(SAM_UNET, self).__init__()
        
        transformer_dim = 256
        img_size = config_dict["img_size"]
        
        self.image_encoder = SAMImageEncodeWrapper(ori_sam=ori_sam, fix=True)
        
        embed_dim = self.image_encoder.embed_dim
        num_patchs = config_dict["img_size"] // self.image_encoder.patch_size
        
        self.prompt_encoder = SAMPromptEncodeWrapper(ori_sam=ori_sam, fix=False)
        self.mask_decoder = SAMMaskDecoderWrapper_Med(ori_sam=ori_sam)
        
        self.resnet = ResNet(type=resnet_type, is_pretrained=is_resnet_pretrained_or_not)
        channel_list = self.resnet.channel_list
        
        self.global_index = self.image_encoder.global_index
        
        self.adapters_in = nn.ModuleList([Reshaper(img_size // 4, channel_list[0], num_patchs, embed_dim),       # [b, channel_list[0], S/4 (64), S/4 (64)] --> [b, num_patches (16), num_patches (16), dim]
                                          Reshaper(img_size // 8, channel_list[1], num_patchs, embed_dim),       # [b, channel_list[1], S/8 (32), S/8 (32)] --> [b, num_patches (16), num_patches (16), dim]                                   # [b, 2048, S/32 (8), S/32 (8)] --> [b, num_patches (16), num_patches (16), dim]
                                          Reshaper(img_size // 16, channel_list[2], num_patchs, embed_dim),      # [b, channel_list[2], S/16 (16), S/16 (16)] --> [b, num_patches (16), num_patches (16), dim]         
                                          Reshaper(img_size // 32, channel_list[3], num_patchs, embed_dim),])    # [b, channel_list[3], S/32 (8), S/32 (8)] --> [b, num_patches (16), num_patches (16), dim]
        
        self.adapters_bridge = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(embed_dim, transformer_dim, kernel_size=2, stride=2),
                LayerNorm2d(transformer_dim),
                nn.GELU(),
                nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
            for _ in range(3)]) # 2 -- top 0 -- bottom
        
        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )
        
        
    @torch.no_grad()
    def infer(self, list_input: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor [TODO] is
        recommended over calling the model directly.

        Arguments:
          list_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'labels': The ground truth mask as a torch tensor in 1x1xHxW format,
                where (H, W) is the original size of the image.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'boxes': (torch.Tensor) Box inputs, with shape 1x4.
                Already transformed to the input frame of the model.
          resnet (nn.Module): A resnet model to extract features from the input.
          global_prompt (bool): Whether to use a global prompt for all images.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape 1x1xHxW, (H, W) is the original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape 1x1.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape 1x1xHxW, where H=W=config_dict["img_size"]//4. 
                Can be passed as mask input to subsequent iterations of prediction.
        """
        
        input_images = torch.stack([preprocess(x["image"], 
                                               pixel_mean = config_dict["pixel_mean"], 
                                               pixel_std = config_dict["pixel_std"]) for x in list_input], dim=0)
        
        feature_maps_in = self.resnet(input_images)                 # a list of feature maps for encoder
        feature_maps_out = [None] * 3                               # a list of feature maps for decoder
        input_images = self.image_encoder.forward_patch_embed(input_images)
        for i in range(len(self.global_index)):
            for j in range(2):
                input_images = self.image_encoder.forward_block(input_images, i * 3 + j)
            current_feature_map = self.adapters_in[i](feature_maps_in[i])
            current_feature_map = current_feature_map.permute(0, 2, 3, 1)
            input_images = input_images + current_feature_map
            input_images = self.image_encoder.forward_block(input_images, self.global_index[i])
            if i in range(len(self.global_index) - 1):  # 0, 1, 2
                permuted_input_images = input_images.permute(0, 3, 1, 2)
                current_out_feature_map = self.adapters_bridge[3 - i - 1](permuted_input_images) # 3 - i - 1 means 2, 1, 0, because the model is a U-Net
                feature_maps_out[3 - i - 1] = current_out_feature_map
        image_embeddings = self.image_encoder.forward_neck(input_images)
        
        self.multi_scale_feature = self.embedding_encoder(image_embeddings)
        self.multi_scale_feature += feature_maps_out[0]
        self.multi_scale_feature += feature_maps_out[1]
        self.multi_scale_feature += feature_maps_out[2]

        outputs = []
        for image_record, curr_embedding, multi_scale_feature in zip(list_input, image_embeddings, self.multi_scale_feature):
            if image_record.get("point_coords", None) is not None:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multi_scale_feature=multi_scale_feature.unsqueeze(0),
            )
            
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            
            outputs.append(
                {
                    "masks": masks,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs
    
    
    def train_forward(self,
                      list_input: List[Dict[str, Any]],
                      ) -> List[Dict[str, torch.Tensor]]:
        """
        Forward funtion for training.
        
        Arguments:
          list_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'labels': The ground truth mask as a torch tensor in num_masksx1xHxW format,
                where H=W=config_dict["img_size"].
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Point prompts for
                this image, with shape num_masksxNxnum_pointsx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Labels for point prompts,
                with shape num_masksxnum_points.
              'boxes': (torch.Tensor) Box inputs, with shape num_masksx4.
                Already transformed to the input frame of the model.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.
          resnet (nn.Module): A resnet model to extract features from the input.
          global_prompt (bool): Whether to use a global prompt for all images.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'logits_masks': (torch.Tensor) Batched logits mask predictions,
                with shape num_masksx1xHxW, where H=W=config_dict["img_size"].
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape num_masksx1.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape num_masksx1xHxW, where H=W=config_dict["img_size"]//4. 
                Can be passed as mask input to subsequent iterations of prediction.
        """

        input_images = torch.stack([preprocess(x["image"], 
                                               pixel_mean = config_dict["pixel_mean"], 
                                               pixel_std = config_dict["pixel_std"]) for x in list_input], dim=0)
        
        feature_maps_in = self.resnet(input_images)                 # a list of feature maps for encoder
        feature_maps_out = [None] * 3                               # a list of feature maps for decoder                 
        input_images = self.image_encoder.forward_patch_embed(input_images)
        for i in range(len(self.global_index)):
            for j in range(2):
                input_images = self.image_encoder.forward_block(input_images, i * 3 + j)
            current_feature_map = self.adapters_in[i](feature_maps_in[i])
            current_feature_map = current_feature_map.permute(0, 2, 3, 1)
            input_images = input_images + current_feature_map
            input_images = self.image_encoder.forward_block(input_images, self.global_index[i])
            if i in range(len(self.global_index) - 1):  # 0, 1, 2
                permuted_input_images = input_images.permute(0, 3, 1, 2)
                current_out_feature_map = self.adapters_bridge[3 - i - 1](permuted_input_images) # 3 - i - 1 means 2, 1, 0, because the model is a U-Net
                feature_maps_out[3 - i - 1] = current_out_feature_map
        image_embeddings = self.image_encoder.forward_neck(input_images)
        
        self.multi_scale_feature = self.embedding_encoder(image_embeddings)
        self.multi_scale_feature += feature_maps_out[0]
        self.multi_scale_feature += feature_maps_out[1]
        self.multi_scale_feature += feature_maps_out[2]

        outputs = []
        for image_record, curr_embedding, multi_scale_feature in zip(list_input, image_embeddings, self.multi_scale_feature):
            if image_record.get("point_coords", None) is not None:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multi_scale_feature=multi_scale_feature.unsqueeze(0),
            )
            
            masks = self.postprocess_training(low_res_masks)
            
            outputs.append(
                {
                    "masks": masks,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs


    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in num_masksx1xHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in num_masksx1xHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks


    def postprocess_training(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Only upscale masks to [num_masks, 1, config_dict["img_size"], config_dict["img_size"]],
        for easily calculating the loss.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return masks  
    
    def forward(self, list_input: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        if self.training:
            return self.train_forward(list_input)
        else:
            return self.infer(list_input)
