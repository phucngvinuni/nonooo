import math
import numpy as np
from timm.models.registry import register_model
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

import torch
from channel import *
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.relu = nn.ReLU()
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = self.relu(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)) #math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #[1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
        #self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)
        
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)
        # print(key.shape)
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        x, self.attn = self.attention(query, key, value, mask=mask)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.num_heads * self.d_k)
             
        x = self.dense(x)
        x = self.dropout(x)
        
        return x
    
    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        #print(mask.shape)
        if mask is not None:
            scores += (mask * -1e9)
            # attention weights
        p_attn = F.softmax(scores, dim = -1)
        return torch.matmul(p_attn, value), p_attn

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x) 
        return x

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout = 0.1)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        #self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        "Follow Figure 1 (right) for connections."
        attn_output = self.self_mha(x, x, x, look_ahead_mask)
        x = self.layernorm1(x + attn_output)
     
        src_output = self.src_mha(x, memory, memory, trg_padding_mask) # q, k, v
        x = self.layernorm2(x + src_output)
        
        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x


class Decoder(nn.Module):
    def __init__(self, depth=4, embed_dim=128, num_heads=4, dff=128, drop_rate=0.1):
        super(Decoder, self).__init__()
    
        self.d_model = embed_dim
        self.pos_encoding = PositionalEncoding(embed_dim, drop_rate, 50)
        self.dec_layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, dff, drop_rate) 
                                            for _ in range(depth)])
        
    def forward(self, x, memory, look_ahead_mask=None, trg_padding_mask=None):
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)  
        return x


class ViTEncoder_FIM(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_fim = 1
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth-self.num_fim)])
        
        self.blocks_fim = nn.ModuleList([
            FIM_V1(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth-1], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(self.num_fim)])
        
        self.norm =  norm_layer(embed_dim)
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, bm_pos, y):
        cls_wise_output = []
        x = self.patch_embed(x)
        B,_,C = x.shape
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        x_vis = x[~bm_pos].reshape(B, -1, C)
        for blk in self.blocks:
            x_vis = blk(x_vis)
        for blk in self.blocks_fim:
            x_vis, class_output = blk(x_vis, bm_pos, y)
            cls_wise_output.append(class_output)    
        x_vis = self.norm(x_vis)
        return x_vis, cls_wise_output

    def forward(self, x, bm_pos, y=None):
        x, cls_wise_output = self.forward_features(x, bm_pos, y)
        return x, cls_wise_output


class ViTEncoder_Van(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, bm_pos):
        x = self.patch_embed(x)
        B,_,C = x.shape
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        x_vis = x[~bm_pos].reshape(B, -1, C)
        for blk in self.blocks:
            x_vis = blk(x_vis)    
        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, bm_pos):
        x = self.forward_features(x, bm_pos)
        return x

class ViTDecoder_Van(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196,
                 ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, return_token_num=0):
        for blk in self.blocks:
            x = blk(x)
        return x


def noise_gen(is_train):
    min_snr, max_snr = -10, 30
    diff_snr = max_snr - min_snr
    
    min_var, max_var = 10**(-min_snr/20), 10**(-max_snr/20)
    diff_var = max_var - min_var
    if is_train:
        # b = torch.bernoulli(1/5.0*torch.ones(1))
        # if b > 0.5:
        #     channel_snr = torch.FloatTensor([20])
        # else:               
        #     channel_snr = torch.rand(1)*diff_snr+min_snr
        # noise_var = 10**(-channel_snr/20)
        # noise_var = torch.rand(1)*diff_var+min_var  
        # channel_snr = 10*torch.log10((1/noise_var)**2)
        # channel_snr = torch.rand(1)*diff_snr+min_snr
        # noise_var = 10**(-channel_snr/20)
        channel_snr = torch.FloatTensor([-10])
        noise_var = torch.FloatTensor([1]) * 10**(-channel_snr/20)  
    else:
        channel_snr = torch.FloatTensor([-10])
        noise_var = torch.FloatTensor([1]) * 10**(-channel_snr/20)  
    return channel_snr, noise_var 
 

class FIM_V2(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.dim_features = 384
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None
        self.fc = nn.Linear(self.dim_features, 10)
        self.relu = nn.ReLU()

    def forward(self, x, maskpos, label=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = self.relu(x)
        B, L, E = x.shape
      
        #print(B,L,E)
        fc_in = torch.mean(x.view(x.shape[0], -1, x.shape[2]), dim=-2)
        fc_out = self.fc(fc_in.view(x.shape[0], x.shape[2]))
        if self.training:
            B, L, E = x.shape
            mask = self.fc.weight[label, :]
            x = x * (mask.view(B, self.dim_features, 1).view(B, 1, self.dim_features))
        else:
            B, L, E = x.shape
            pred_label = torch.max(fc_out, dim=1)[1]
            mask = self.fc.weight[pred_label, :]
            x = x * (mask.view(B, self.dim_features, 1).view(B, 1, self.dim_features))
        
        return x, fc_out


class FIM_V1(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.num_features = 196
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None
        self.fc = nn.Linear(self.num_features, 10)
        self.relu = nn.ReLU()
    def forward(self, x, maskpos, label=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = self.relu(x)
      
        
        B, L, E = x.shape
        temp = torch.zeros(B,self.num_features,E,dtype=x.dtype).cuda()
        temp[~maskpos] = x.reshape(B*L, E)
        temp = temp.reshape(B,self.num_features,E)
        #print(B,L,E)
        fc_in = torch.mean(temp.view(temp.shape[0], temp.shape[1], -1), dim=-1)
        fc_out = self.fc(fc_in.view(temp.shape[0], temp.shape[1]))
        if self.training:
            B, L, E = x.shape
            mask = self.fc.weight[label, :]
            print(f"mask.shape: {mask.shape}")
            x = x * mask.view(B, L, 1)  # [50, 196, 1]
        else:
            B, L, E = x.shape
            pred_label = torch.max(fc_out, dim=1)[1]
            mask = self.fc.weight[pred_label, :]

            print(f"mask.shape: {mask.shape}")
            x = x * mask.view(B, L, 1)  # [50, 196, 1]
 
        # print(print(x.mean(-1)).shape)
        return x, fc_out
    def forward(self, x, maskpos, label=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        x = self.relu(x)
        
        B, L, E = x.shape
        temp = torch.zeros(B, self.num_features, E, dtype=x.dtype).cuda()
        temp[~maskpos] = x.reshape(B * L, E)
        temp = temp.reshape(B, self.num_features, E)
        fc_in = torch.mean(temp.view(temp.shape[0], temp.shape[1], -1), dim=-1)
        fc_out = self.fc(fc_in.view(temp.shape[0], temp.shape[1]))
        
        if self.training:
            assert label is not None, "Label must be provided during training"
            assert label.shape[0] == B, f"Label shape {label.shape} must match batch size {B}"
            mask = self.fc.weight[label, :]  # [B, 196]
            if mask.shape[1] != L:
                mask = mask[:, :L]  # Adjust to match x's sequence length: [B, L]
            # print(f"Training mask.shape after adjust: {mask.shape}")
            x = x * mask.view(B, L, 1)  # [B, L, 1]
        else:
            pred_label = torch.max(fc_out, dim=1)[1]  # [B]
            mask = self.fc.weight[pred_label, :]  # [B, 196]
            if mask.shape[1] != L:
                mask = mask[:, :L]  # Adjust to match x's sequence length: [B, L]
            # print(f"Inference mask.shape after adjust: {mask.shape}")
            x = x * mask.view(B, L, 1)  # [B, L, 1]
        
        return x, fc_out



# class VectorQuantizer(nn.Module):
#     """
#     Reference:
#     [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
#     """
#     def __init__(self,
#                  num_embeddings: int,
#                  embedding_dim: int,
#                  beta: float = 0.25):
#         super(VectorQuantizer, self).__init__()
#         self.K = num_embeddings
#         self.D = embedding_dim
#         self.beta = beta

#         self.embedding = nn.Embedding(self.K, self.D)
#         self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

#     def forward(self, latents, SNRdB, bit_per_index):
#         latents = latents # [B x L x D]
#         latents_shape = latents.shape
#         flat_latents = latents.view(-1, self.D)  # [BL x D]
#         device = latents.device
#         # Compute L2 distance between latents and embedding weights
#         dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
#                torch.sum(self.embedding.weight ** 2, dim=1) - \
#                2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BL x K]

#         # Get the encoding that has the min distance
#         encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BL, 1]
#         shape = encoding_inds.shape
#         Rx_signal = transmit(encoding_inds, SNRdB, bit_per_index)
#         encoding_inds = torch.from_numpy(Rx_signal).to(device).reshape(shape)
        
#         # Convert to one-hot encodings   
#         encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
#         encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BL x K]

#         # Quantize the latents
#         quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BL, D]
#         quantized_latents = quantized_latents.view(latents_shape)  # [B x L x D]

#         # Compute the VQ Losses
#         commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
#         embedding_loss = F.mse_loss(quantized_latents, latents.detach())

#         vq_loss = commitment_loss * self.beta + embedding_loss

#         # Add the residue back to the latents
#         quantized_latents = latents + (quantized_latents - latents).detach()
#         return quantized_latents.contiguous(), vq_loss  # [B x L x D]



class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents, SNRdB, bit_per_index):
        latents = latents  # [B x L x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BL x D]
        device = latents.device
        
        # 计算与码本的 L2 距离
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BL x K]
        
        # 找到最近的码本索引
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BL, 1]
        
        # 转换为 one-hot 编码
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BL x K]
        
        # 量化特征
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BL, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x L x D]
        
        # 计算 VQ 损失
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        vq_loss = commitment_loss * self.beta + embedding_loss
        
        # 保留梯度传递
        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents.contiguous(), vq_loss

class Channels():
    
    def AWGN(self, Tx_sig, n_var):
        device = Tx_sig.device
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        device = Tx_sig.device
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        device = Tx_sig.device
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans  # Sử dụng scikit-learn cho k-means

class KMeansQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25, kmeans_init=True, kmeans_iters=10):
        super().__init__()
        self.num_embeddings = num_embeddings  # Số lượng centroids (codebook size)
        self.embedding_dim = embedding_dim    # Chiều của embedding
        self.commitment_cost = commitment_cost # Hệ số cho commitment loss
        self.kmeans_init = kmeans_init       # Khởi tạo centroids bằng k-means++?
        self.kmeans_iters = kmeans_iters       # Số lần lặp tối đa của k-means

        # Khởi tạo embedding (codebook)
        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, x):
        # x: [B, ..., embedding_dim]  (B: batch size, ...: các chiều khác)

        # Flatten đầu vào
        x_flat = x.view(-1, self.embedding_dim)  # [N, embedding_dim] (N = B * ...)

        # Khởi tạo centroids bằng k-means++ (nếu cần)
        if self.kmeans_init:
            kmeans = KMeans(n_clusters=self.num_embeddings, init='k-means++', max_iter=self.kmeans_iters, n_init=1)
            kmeans.fit(x_flat.detach().cpu().numpy()) #fit tren cpu
            self.embedding.data.copy_(torch.from_numpy(kmeans.cluster_centers_).to(x.device)) #.to(x.device))
            self.kmeans_init = False  # Chỉ khởi tạo một lần

        # Tính khoảng cách Euclidean từ mỗi điểm đến các centroids
        distances = (
            torch.sum(x_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embedding**2, dim=1)
            - 2 * torch.matmul(x_flat, self.embedding.t())
        )  # [N, num_embeddings]

        # Tìm centroid gần nhất (argmin)
        encoding_indices = torch.argmin(distances, dim=1)  # [N]

        # One-hot encoding
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)  # [N, num_embeddings]

        # Quantization
        quantized = torch.matmul(encodings, self.embedding)  # [N, embedding_dim]
        quantized = quantized.view(x.shape)  # Reshape lại kích thước ban đầu

        # Commitment loss (để encoder tạo ra các vector gần với centroids)
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach()) # q loss
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator (copy gradient)
        quantized = x + (quantized - x).detach()

        # Tính perplexity (độ phức tạp, đo lường mức độ sử dụng codebook)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity, encoding_indices