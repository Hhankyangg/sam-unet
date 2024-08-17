import torch
from torch import nn

from typing import Tuple
from sam_unet.models.segment_anything.modeling.common import LayerNorm2d
from sam_unet.models.segment_anything.modeling.mask_decoder import MLP

class SAMImageEncodeWrapper(nn.Module):

    def __init__(self, ori_sam, fix: bool = True):
        
        super(SAMImageEncodeWrapper, self).__init__()
        
        self.sam_img_encoder = ori_sam.image_encoder
        
        self.patch_size = self.sam_img_encoder.patch_size
        self.depth = self.sam_img_encoder.depth
        self.prompt_dim = self.sam_img_encoder.embed_dim
        self.embed_dim = self.sam_img_encoder.embed_dim
        self.img_size = self.sam_img_encoder.img_size
        self.global_index = self.sam_img_encoder.global_index
        
        if fix:
            for name, param in self.sam_img_encoder.named_parameters():
                param.requires_grad = False


    def forward(self, x, prompt_tokens: torch.Tensor = None):
        
        # prompt_tokens [b, depth, num_prompts, prompt_dim]
        x = self.sam_img_encoder.patch_embed(x)
        if self.sam_img_encoder.pos_embed is not None:
            x = x + self.sam_img_encoder.pos_embed
            
        for idx, blk in enumerate(self.sam_img_encoder.blocks):
            current_prompt = prompt_tokens[:, idx, :, :] if prompt_tokens is not None else None
            # current_prompt [b, num_prompts, prompt_dim]
            x = blk(x, prompt_tokens=current_prompt)
            
        x = self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))
        return x
    
    
    def forward_patch_embed(self, x):
        x = self.sam_img_encoder.patch_embed(x)
        if self.sam_img_encoder.pos_embed is not None:
            x = x + self.sam_img_encoder.pos_embed
        return x
    
    
    def forward_block(self, x, idx):
        x = self.sam_img_encoder.blocks[idx](x)
        return x
    
    
    def forward_neck(self, x):
        x = self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))
        return x


class SAMPromptEncodeWrapper(nn.Module):

    def __init__(self, ori_sam, fix: bool = True):
        super(SAMPromptEncodeWrapper, self).__init__()
        self.sam_prompt_encoder = ori_sam.prompt_encoder
        if fix:
            for name, param in self.sam_prompt_encoder.named_parameters():
                param.requires_grad = False

    def forward(self, points=None, boxes=None, masks=None):
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points, boxes, masks)
        return sparse_embeddings, dense_embeddings

    def get_dense_pe(self):
        return self.sam_prompt_encoder.get_dense_pe()
    

class SAMMaskDecoderWrapper_Med(nn.Module):

    def __init__(self, ori_sam, transformer_dim: int = 256):
        super(SAMMaskDecoderWrapper_Med, self).__init__()
        self.sam_mask_decoder = ori_sam.mask_decoder
        
        self.med_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multi_scale_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multi_scale_feature = multi_scale_feature,
        )

        # Prepare output
        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multi_scale_feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
      
        hq_token_weight = self.med_token.weight

        # Concatenate output tokens
        output_tokens = hq_token_weight
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.sam_mask_decoder.transformer(src, pos_src, tokens)
        mask_tokens_out = hs[:, 0, :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding_sam = self.sam_mask_decoder.output_upscaling(src)
        upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding_sam) + multi_scale_feature.repeat(b, 1, 1, 1)
        
        hyper_in = self.hf_mlp(mask_tokens_out).unsqueeze(1)
        b, c, h, w = upscaled_embedding_sam.shape
        masks_hq = (hyper_in @ upscaled_embedding_hq.view(b, c, h * w)).view(b, -1, h, w)

        return masks_hq
