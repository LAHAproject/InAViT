import math
from functools import partial
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from collections import OrderedDict

import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.attention import MultiScaleBlock, TrajectoryAttentionBlock, EntropyPrunedSelfAttention, SelfAttentionBlock, CrossAttentionBlock
from slowfast.models.attention import CrossAttention, TrajectoryAttention, SelfAttentionBlock, SelfAttention, TrajectoryCrossAttention
from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.stem_helper import PatchEmbed
from slowfast.models.utils import round_width
from . import ObjectsCrops
from .orvit import DropPath, MotionStream, drop_path

from slowfast.datasets.utils import mask_fg

from slowfast.models import head_helper, resnet_helper, stem_helper
from slowfast.models.build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class OnlyHOIViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, cfg):
        super().__init__()

        self.img_size = cfg.DATA.TRAIN_CROP_SIZE
        self.patch_size = cfg.MF.PATCH_SIZE
        self.in_chans = cfg.MF.CHANNELS
        if cfg.TRAIN.DATASET == "epickitchens":
            self.num_classes = [97, 300]  
        else:
            self.num_classes = cfg.MODEL.NUM_CLASSES
        self.embed_dim = cfg.MF.EMBED_DIM
        self.depth = cfg.MF.DEPTH
        self.num_heads = cfg.MF.NUM_HEADS
        self.mlp_ratio = cfg.MF.MLP_RATIO
        self.qkv_bias = cfg.MF.QKV_BIAS
        self.drop_rate = cfg.MF.DROP
        self.drop_path_rate = cfg.MF.DROP_PATH
        self.head_dropout = cfg.MF.HEAD_DROPOUT
        self.video_input = cfg.MF.VIDEO_INPUT
        self.temporal_resolution = cfg.MF.TEMPORAL_RESOLUTION
        self.use_mlp = cfg.MF.USE_MLP
        self.num_features = cfg.MF.EMBED_DIM
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.attn_drop_rate = cfg.MF.ATTN_DROPOUT
        self.head_act = cfg.MF.HEAD_ACT
        self.entropy_threshold = cfg.MASKEDHOIVIT.ENTROPY_THRESHOLD
        self.decay_rate = cfg.MASKEDHOIVIT.DECAY_RATE
        self.with_motion_stream = cfg.MASKEDHOIVIT.USE_MOTION_STREAM
        self.cfg = cfg
        dim = self.embed_dim
        
        
        self.patch_embed_3d = stem_helper.PatchEmbed(
            dim_in=self.in_chans,
            dim_out=dim,
            kernel=[self.cfg.MF.PATCH_SIZE_TEMP, self.patch_size, self.patch_size], 
            stride=[self.cfg.MF.PATCH_SIZE_TEMP, self.patch_size, self.patch_size],
            padding=0,
            conv_2d=False,
        )

        self.patch_embed_3d.num_patches = (224 // self.patch_size) ** 2
        self.patch_embed_3d.proj.weight.data = torch.zeros_like(
            self.patch_embed_3d.proj.weight.data)
        # Number of patches
        if self.video_input:
            num_patches = self.patch_embed_3d.num_patches * self.temporal_resolution
        else:
            num_patches = self.patch_embed_3d.num_patches
        self.num_patches = num_patches

        # CLS token
        self.with_cls_token = True 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        trunc_normal_(self.cls_token, std=.02)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed_3d.num_patches + 1, dim))
        self.pos_drop = nn.Dropout(p=cfg.MF.POS_DROPOUT)
        trunc_normal_(self.pos_embed, std=.02)

        if self.cfg.MF.POS_EMBED == "joint":
            self.st_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, dim))
            trunc_normal_(self.st_embed, std=.02)
        elif self.cfg.MF.POS_EMBED == "separate":
            self.temp_embed = nn.Parameter(
                torch.zeros(1, self.temporal_resolution, dim))

        # Layer Blocks
        dpr = [x.item() for x in torch.linspace(
            0, self.drop_path_rate, self.depth)]
        
        self.box_categories = nn.Parameter(torch.zeros(self.temporal_resolution, self.cfg.HOIVIT.O + self.cfg.HOIVIT.U, dim))
        self.c_coord_to_feature = nn.Sequential(
            nn.Linear(4, dim // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim, bias=False),
            nn.ReLU()
        )

        ##
        blocks = []
        for i in range(self.depth-1):
            _block = TrajectoryAttentionBlock(
                    cfg = cfg,
                    dim=dim, 
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio, 
                    qkv_bias=self.qkv_bias, 
                    drop=self.drop_rate, 
                    attn_drop=self.attn_drop_rate, 
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
            )
            blocks.append(_block)
        self.blocks = nn.ModuleList(blocks)

        self.norm = norm_layer(dim)

        # MLP head
        if self.use_mlp:
            hidden_dim = dim
            if self.head_act == 'tanh':
                print("Using TanH activation in MLP")
                act = nn.Tanh() 
            elif self.head_act == 'gelu':
                print("Using GELU activation in MLP")
                act = nn.GELU()
            else:
                print("Using ReLU activation in MLP")
                act = nn.ReLU()
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim, hidden_dim)),
                ('act', act),
            ]))
        else:
            self.pre_logits = nn.Identity()
        
        # Classifier Head
        self.head_drop = nn.Dropout(p=self.head_dropout)
        if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
            for a, i in enumerate(range(len(self.num_classes))):
                setattr(self, "head%d"%a, nn.Linear(dim, self.num_classes[i]))
        else:
            self.head = (nn.Linear(dim, self.num_classes) 
                if self.num_classes > 0 else nn.Identity())

        # Initialize weights
        self.init_weights()
        self.apply(self._init_weights)

        # Object Tokens
        self.crop_layer = ObjectsCrops(cfg)
        self.patch_to_d = nn.Sequential(
            nn.Linear(dim, dim // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim, bias=False),
            nn.ReLU()
        )

        self.prunedattn = EntropyPrunedSelfAttention(
            dim=dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            proj_drop=self.drop_rate,
            entropy_threshold=self.entropy_threshold,
            decay_rate=self.decay_rate
        )

        self.spatial_crossattn = CrossAttention(
            dim=dim, 
            num_heads=self.num_heads, 
            qkv_bias=self.qkv_bias,
            proj_drop=self.drop_rate
        )

        # Motion Stream Attention Block parameters
        mlp_ratio = cfg.MF.MLP_RATIO
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        drop_path = dpr[1]
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        act_layer=nn.GELU
               
        
    def init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cfg.MF.POS_EMBED == "joint":
            return {'pos_embed', 'cls_token', 'st_embed'}
        else:
            return {'pos_embed', 'cls_token', 'temp_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = (nn.Linear(self.embed_dim, num_classes) if num_classes > 0
            else nn.Identity())

    def forward_features(self, x, obj_box_tensors, hand_box_tensors): # x: [BS, C=3, T=16, H=224, W=224]
        if self.video_input:
            x = x[0]
        B = x.shape[0]
        (B, C,T, H, W) = x.shape
        
        # Tokenize input
        x = self.patch_embed_3d(x) # [BS, N=T'*H'*W', d]

        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # [BS, 1, dim]
        x = torch.cat((cls_tokens, x), dim=1) # [BS, N, dim]

        # Interpolate positinoal embeddings
        if self.cfg.DATA.TRAIN_CROP_SIZE != 224:
            pos_embed = self.pos_embed
            N = pos_embed.shape[1] - 1
            npatch = int((x.size(1) - 1) / self.temporal_resolution)
            class_emb = pos_embed[:, 0]
            pos_embed = pos_embed[:, 1:]
            dim = x.shape[-1]
            pos_embed = torch.nn.functional.interpolate(
                pos_embed.reshape(
                    1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                    0, 3, 1, 2),
                scale_factor=math.sqrt(npatch / N),
                mode='bicubic',
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            new_pos_embed = torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)
        else:
            new_pos_embed = self.pos_embed
            npatch = self.patch_embed_3d.num_patches

        # Add positional embeddings to input
        if self.video_input:
            if self.cfg.MF.POS_EMBED == "separate":
                cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
                tile_pos_embed = new_pos_embed[:, 1:, :].repeat(
                    1, self.temporal_resolution, 1)
                tile_temporal_embed = self.temp_embed.repeat_interleave(
                    npatch, 1)
                total_pos_embed = tile_pos_embed + tile_temporal_embed
                total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)
                x = x + total_pos_embed
            elif self.cfg.MF.POS_EMBED == "joint":
                x = x + self.st_embed
        else:
            # image input
            x = x + new_pos_embed
                            
        # Apply positional dropout
        x = self.pos_drop(x) # [BS, N, dim]
        # Encoding using transformer layers
        thw = [self.temporal_resolution, int(npatch**0.5), int(npatch**0.5)]
        
        return x, thw

    def forward(self, x, metadata): # x: [BS, C=3, T=16, H=224, W=224]
        
        
        obj_box_tensors = metadata['orvit_bboxes']['obj']
        assert obj_box_tensors is not None
        hand_box_tensors = metadata['orvit_bboxes']['hand']
        assert hand_box_tensors is not None

        x, thw = self.forward_features(x, obj_box_tensors, hand_box_tensors) # [BS, d]

        if self.with_cls_token:
            cls_token, patch_tokens = x[:,[0]], x[:,1:]

        BS, _, d = x.shape
        T,H,W = thw
        patch_tokens = patch_tokens.permute(0,2,1).reshape(BS, -1, T,H,W).detach()
        
        BS, d, T, H, W = patch_tokens.shape
        assert T == self.temporal_resolution

        Tratio = obj_box_tensors.shape[1] // T
        
        obj_box_tensors = obj_box_tensors[:,::Tratio] # [BS, T , O, 4]
        hand_box_tensors = hand_box_tensors[:,::Tratio] # [BS, T , U, 4]
        
        # handle if hand_box_tensors is missing a dimension as it is only a single box
        if len(hand_box_tensors.shape) == 3:
            hand_box_tensors = hand_box_tensors.unsqueeze(-2) # convert [BS, T, 4] -> [BS, T, 1, 4]
        
        O = obj_box_tensors.shape[-2]
        U = hand_box_tensors.shape[-2]
        
        
        obj_tokens = self.crop_layer(patch_tokens, obj_box_tensors)  # [BS, O,T, d, H, W]
        obj_tokens = obj_tokens.permute(0, 1,2,4,5,3)  # [BS, O,T, H, W, d]
        obj_tokens = self.patch_to_d(obj_tokens) # [BS,O,T, H, W, d]
        obj_tokens = obj_tokens.reshape(BS, T, O*H*W, d) 
        # obj_tokens = torch.amax(obj_tokens, dim=(-3,-2)) # [BS, O,T, d]
        # obj_tokens = obj_tokens.permute(0,2,1,3)

        hand_tokens = self.crop_layer(patch_tokens, hand_box_tensors)  # [BS, U,T, d, H, W]
        hand_tokens = hand_tokens.permute(0, 1,2,4,5,3)  # [BS, U,T, H, W, d]
        hand_tokens = self.patch_to_d(hand_tokens) # [BS,U,T, H, W, d]
        hand_tokens = hand_tokens.reshape(BS, T, U*H*W, d)
        # hand_tokens = torch.amax(hand_tokens, dim=(-3,-2)) # [BS, U,T, d]
        # hand_tokens = hand_tokens.permute(0,2,1,3)

        hand_tokens_mod, obj_tokens_mod = [], []
        for t in range(T):
            hand_tokens_frame = hand_tokens[:,t,:,:]
            obj_tokens_frame = obj_tokens[:,t,:,:]
            hand_obj_tokens_frame = torch.cat((hand_tokens_frame, obj_tokens_frame),dim=1)
            hand_tokens_frame = self.spatial_crossattn(hand_tokens_frame, obj_tokens_frame)
            #obj_tokens_frame = self.spatial_crossattn(obj_tokens_frame, hand_tokens_frame)
            obj_tokens_frame = self.spatial_crossattn(obj_tokens_frame, hand_obj_tokens_frame) # take object object interactions into account
            
            hand_tokens_mod.append(hand_tokens_frame)
            obj_tokens_mod.append(obj_tokens_frame)
        hand_tokens_mod = torch.cat(hand_tokens_mod).reshape(BS, T, U*H*W, d)
        obj_tokens_mod = torch.cat(obj_tokens_mod).reshape(BS, T, O*H*W, d)
        hand_obj_tokens = torch.cat([obj_tokens_mod, hand_tokens_mod], dim=2)

        box_categories = self.box_categories.unsqueeze(0).expand(BS,-1,-1,-1)
        box_emb = self.c_coord_to_feature(torch.cat([obj_box_tensors, hand_box_tensors], dim=2).float())
        hand_obj_tokens = hand_obj_tokens + box_categories + box_emb # [BS, T, O+U, d]

        thw_new = [T, hand_obj_tokens.shape[2], 1]

        hand_obj_tokens = hand_obj_tokens.flatten(1,2) # [BS, T * (O+U),d]
               
        if self.with_cls_token:
            hand_obj_tokens =  torch.cat([cls_token, hand_obj_tokens], dim = 1) # [BS, 1 + T*N, d]

        for i, blk in enumerate(self.blocks):
            hand_obj_tokens, _ = blk(
                hand_obj_tokens,
                metadata,
                thw_new,
            )

        hand_obj_tokens = self.norm(hand_obj_tokens)[:, 0]
        hand_obj_tokens = self.pre_logits(hand_obj_tokens)
        if not torch.isfinite(hand_obj_tokens).all():
            print("WARNING: nan in features out")
        
        hand_obj_tokens = self.head_drop(hand_obj_tokens)
        if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
            output = []
            for head in range(len(self.num_classes)):
                x_out = getattr(self, "head%d"%head)(hand_obj_tokens)
                if not self.training:
                    x_out = torch.nn.functional.softmax(x_out, dim=-1)
                output.append(x_out)
            return output[0], {'verb': output[0], 'noun': output[1]}
        else:
            hand_obj_tokens = self.head(hand_obj_tokens)
            if not self.training:
                hand_obj_tokens = torch.nn.functional.softmax(hand_obj_tokens, dim=-1)
            return hand_obj_tokens



