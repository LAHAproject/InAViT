import torch
from torch import nn
import torch.nn.functional as F

from slowfast.datasets.utils import mask_fg

from .utils import ObjectsCrops, box2spatial_layout, Mlp

from slowfast.models.attention import CrossAttention, TrajectoryAttention, SelfAttentionBlock, SelfAttention, TrajectoryCrossAttention

from .orvit import MotionStream

import slowfast.utils.logging as logging

from slowfast.models.InAViT.utils import get_union_box

logger = logging.get_logger(__name__)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class UNIONHOIVIT(nn.Module):

    def __init__(
            self, cfg, dim=768, dim_out=None, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code = False, nb_frames=None,
        ):
        super().__init__()

        self.cfg = cfg
        self.in_dim = dim
        self.dim = dim
        self.nb_frames = nb_frames
        
        self.with_cls_token = True 
        self.with_motion_stream = cfg.UNIONHOIVIT.USE_MOTION_STREAM

        self.maskfg = cfg.UNIONHOIVIT.MASK_FG

        # Object Tokens
        self.crop_layer = ObjectsCrops(cfg)
        self.patch_to_d = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim // 2, self.dim, bias=False),
            nn.ReLU()
        )

        self.box_categories = nn.Parameter(torch.zeros(self.nb_frames, self.cfg.HOIVIT.O + self.cfg.HOIVIT.U, self.in_dim))
        self.c_coord_to_feature = nn.Sequential(
            nn.Linear(4, self.in_dim // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_dim // 2, self.in_dim, bias=False),
            nn.ReLU()
        )

        # Attention Block
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = TrajectoryAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.timecrossattn = TrajectoryCrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.spatial_crossattn = CrossAttention(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias,
            proj_drop=drop
        )

        self.spatial_selfattn = SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=drop
        )
        
        self.selfattn = SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=drop
        )

        if self.with_motion_stream:
            self.motion_stream = MotionStream(cfg, dim=dim, num_heads=num_heads, attn_type=cfg.ORVIT.MOTION_STREAM_ATTN_TYPE, 
                                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, 
                                                drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer,
                                                nb_frames=self.nb_frames,
                                                )
            self.motion_mlp = Mlp(in_features=cfg.ORVIT.MOTION_STREAM_DIM if cfg.ORVIT.MOTION_STREAM_DIM > 0 else dim,
                                        hidden_features=mlp_hidden_dim, out_features=dim,
                                        act_layer=act_layer, drop=drop)

        if self.cfg.ORVIT.INIT_WEIGHTS:
            self.apply(self._init_weights)

        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        else:
           for p in m.parameters(): nn.init.normal_(p, std=0.02)

    def _compute_centers(self, box_tensors):
        left_top = box_tensors[:,:,:, :2] # [BS, T, O/U, 4]
        right_bottom = box_tensors[:,:,:, 2:]
        centers = (right_bottom - left_top) // 2
        return centers

    def forward(self, x, metadata,thw):
        get_patch_orig = False
        if self.maskfg:
            if isinstance(x, tuple):
                x, x_orig = x
                
                get_patch_orig = True
        obj_box_tensors = metadata['orvit_bboxes']['obj']
        assert obj_box_tensors is not None
        hand_box_tensors = metadata['orvit_bboxes']['hand']
        assert hand_box_tensors is not None
        
        if self.with_cls_token:
            cls_token, patch_tokens = x[:,[0]], x[:,1:]

        BS, _, d = x.shape
        T,H,W = thw
        patch_tokens = patch_tokens.permute(0,2,1).reshape(BS, -1, T,H,W)
        
        BS, d, T, H, W = patch_tokens.shape
        assert T == self.nb_frames

        Tratio = obj_box_tensors.shape[1] // T

        # reshape hand_box_tensors according to object_box_tensors
        hand_box_tensors = hand_box_tensors.reshape(BS, obj_box_tensors.shape[1], -1, 4)

        obj_box_tensors = obj_box_tensors[:,::Tratio] # [BS, T , O, 4]
        hand_box_tensors = hand_box_tensors[:,::Tratio] # [BS, T , U, 4]
        
        
        O = obj_box_tensors.shape[-2]
        U = hand_box_tensors.shape[-2]

        
        obj_traj = self._compute_centers(obj_box_tensors).float() # [BS, T, O, 2]
        hand_traj = self._compute_centers(hand_box_tensors).float() # [BS, T, U, 2]
        
        dist = []
        for o in range(O):
            m_u = torch.sum(torch.cdist(hand_traj[:,:,0], obj_traj[:,:,o], p=2), dim=-1)
            if len(m_u.shape) > 1:
                m_u = torch.sum(m_u, dim=-1)
            m_u = 1 / ((m_u // T))
            m_u = torch.nan_to_num(m_u, nan=1.0, posinf=1.0) # replace NaN, inf by 1.0
            dist.append(torch.sum(m_u))
        close_obj_idx = dist.index(min(dist))
        obj_box_tensors = obj_box_tensors[:,:,close_obj_idx,:] #[BS, T, 4]

        # computing union box coordinates
        union_box_tensors = torch.zeros((BS, T, 4))
        for t in range(T):
            union_box_tensor = get_union_box(obj_box_tensors[:,t], hand_box_tensors[:,t,0]) # [BS, 4]
            union_box_tensors[:,t,:] = union_box_tensor
        if torch.cuda.is_available():
            union_box_tensors = union_box_tensors.unsqueeze(2).cuda()
        
        if self.maskfg and get_patch_orig:
                patch_tokens_orig = x_orig.permute(0,2,1).reshape(BS, -1, T,H,W)
                union_tokens = self.crop_layer(patch_tokens_orig, union_box_tensors)  # [BS, T, d, H, W]  
        else:    
            union_tokens = self.crop_layer(patch_tokens, union_box_tensors)  # [BS, T, d, H, W]
        
        union_tokens = union_tokens.squeeze(1) # convert to [BS,T, d, H, W]
        union_tokens = union_tokens.permute(0, 1,3,4,2)  # [BS,T, H, W, d]
        union_tokens = self.patch_to_d(union_tokens) # [BS, T, H, W, d]
        union_tokens =torch.amax(union_tokens, dim=(-3,-2)) # [BS, T, d]
        #union_tokens = union_tokens.permute(0,2,1,3)


        union_tokens, _ = self.selfattn(union_tokens, list(union_tokens.shape)) 
        
        all_tokens = torch.cat([patch_tokens.permute(0,2,3,4,1).reshape(BS, T, H*W, d), union_tokens.unsqueeze(2)], dim = 2).flatten(1,2) # [BS, T * (H*W+O+U),d]
        if self.with_cls_token:
            all_tokens =  torch.cat([cls_token, all_tokens], dim = 1) # [BS, 1 + T*N, d]
            if self.cfg.UNIONHOIVIT.DOUBLEATTENTION:
                patch_tokens = torch.cat([cls_token, patch_tokens.permute(0,2,3,4,1).reshape(BS, T, H*W, d).flatten(1,2)], dim = 1) # [BS, 1 + T*N, d]
                union_tokens = torch.cat([cls_token, union_tokens], dim = 1)  # [BS, 1 + T*(O+U), d]
        
        all_tokens, thw = self.attn(
                    self.norm1(all_tokens), 
                    [T, H*W + 1, 1], 
                )

        if self.with_cls_token:
            cls_token, all_tokens =  all_tokens[:, [0]], all_tokens[:, 1:]

        if self.cfg.UNIONHOIVIT.DOUBLEATTENTION:
            patch_tokens, _ = self.timecrossattn(
                        self.norm2(union_tokens),
                        self.norm1(patch_tokens),
                        [T, H*W, 1]
                    )
            cls_token_new, patch_tokens =  patch_tokens[:, [0]], patch_tokens[:, 1:]
            cls_token = 0.5 * (cls_token + cls_token_new)
            patch_tokens = 0.5 * (patch_tokens + all_tokens.reshape(BS,T,H*W+1,d)[:,:,:H*W].reshape(BS,T*H*W,d))
        else:
            patch_tokens = all_tokens.reshape(BS,T,H*W+1,d)[:,:,:H*W].reshape(BS,T*H*W,d)

        # if self.with_motion_stream:
        #     #box_tensors = torch.cat((obj_box_tensors, hand_box_tensors), dim=-2)
        #     motion_emb = self.motion_stream(union_box_tensors,H, W) # [BS, T, H, W, d]
        #     motion_emb = self.motion_mlp(motion_emb) # [BS, T*H*W, d]
        #     patch_tokens = patch_tokens + motion_emb
        

        if self.with_cls_token:
            patch_tokens = torch.cat([cls_token, patch_tokens], dim = 1) # [BS, 1 + N, d]

        x = x + self.drop_path(patch_tokens) # [BS, N, d]
        x = x + self.drop_path(self.mlp(self.norm2(x))) # [BS, N, d]

        return x, thw

class Object2Spatial(nn.Module):
    def __init__(self, cfg, _type):
        super().__init__()
        self.cfg = cfg
        self._type = _type 
    def forward(self, all_features, context, boxes, H, W, t_avg_pooling = False):
        BS, T, O, d = all_features.shape

        if self._type == 'layout':
            ret = box2spatial_layout(boxes, all_features,H,W) # [B, d, T, H, W]
            ret = ret.permute(0,2,3,4,1)
            if t_avg_pooling:
                BS, T, H, W, d = ret.size()
                Tratio = int(T / self.cfg.MF.TEMPORAL_RESOLUTION)
                if Tratio > 1:
                    ret = ret.reshape(BS, -1, Tratio, H, W, d).mean(2)
            ret = ret.flatten(1,3) # [BS, T*H*W, d]
        elif self._type == 'spatial_only':
            assert context is not None
            ret = context.flatten(1,-2) # [BS, T*H*W, d]
        elif self._type == 'object_pooling':
            ret = torch.amax(all_features, dim = 2) # [BS, T, d]
            ret = ret.reshape(BS, T, 1,1, d).expand(BS, T, H, W, d).flatten(1,3)
        elif self._type == 'all_pooling':
            ret = torch.amax(all_features, dim = [1,2]) # [BS, T, d]
            ret = ret.reshape(BS, 1, 1,1, d).expand(BS, T, H, W, d).flatten(1,3)
        else:
            raise NotImplementedError(f'{self._type}')
        return ret
