__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
import math


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):                                 # x: [b, c, boundary_num_period*boundary_period*d_model]
        x = self.linear(x)
        x = self.dropout(x)
        return x


# Cell
class MGSformer_backbone(nn.Module):
    def __init__(self, configs, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024,
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False,**kwargs):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Head
        self.head_nf = d_model * context_window
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual
        self.seq_len = context_window
        self.period = configs.period
        self.n = configs.n
        self.n_intra = configs.n_intra
        self.traffic = configs.traffic
        
        # period
        if self.seq_len % self.period == 0:
            num_period = self.seq_len//self.period
        else:
            num_period = self.seq_len//self.period + 1
            self.seq_len += self.period-self.seq_len%self.period
            
        # Backbone
        if self.traffic:
            self.backbone = TSTiEncoder(c_in*context_window, c_in, self.period, self.n, self.n_intra, num_period, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        else:
            self.backbone = TSTiEncoder(c_in*configs.compress_len, c_in, self.period, self.n, self.n_intra, num_period, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                        n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                        attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                        attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                        pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        self.head = Flatten_Head(self.individual, self.n_vars, d_model + num_period*d_model + num_period*self.period*d_model, target_window, head_dropout=head_dropout)
        self.compress = nn.Linear(context_window, configs.compress_len)

    def forward(self, z):                                                                   # z: [bs x c x seq_len]
        device = z.device
        # norm
        if self.revin:
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)

        b, c, l = z.size()
        ######################## coarse-scale 
        if self.traffic:
            z_compressed = z
        else:
            z_compressed = self.compress(z)              # [b, c, 10]
        padded_coarse = torch.zeros(b, c, c, z_compressed.shape[-1], device=device)  # padded: [b, num_dynamic, c*10]
        flag = 0
        for u in range(0, c, 1):  # u: [0, num_period-1]
            start_idx1 = u
            end_idx1 = c
            start_idx2 = 0
            end_idx2 = c - u
            padded_coarse[:, flag, start_idx1:end_idx1, :] = z_compressed[:, start_idx2:end_idx2, :]
            flag += 1
        z_coarse = padded_coarse.view(padded_coarse.shape[0], padded_coarse.shape[1], padded_coarse.shape[-2] * padded_coarse.shape[-1])     # [b, c, num_dynamic*10]
        # model
        z_coarse = self.backbone(z_coarse, 'coarse')                                        # z: [b, c, d_model]
        
        ######################## middle-scale 
        period = self.period
        # period
        if l % period == 0:
            num_period = l//period
        else:
            num_period = l//period + 1
            z = torch.nn.functional.pad(z, (period - l % period, 0))
        z = z.view(b, c, num_period, period)         # z: [b, c, num_period, period]        
        padded_intra = torch.zeros(z.shape[0], z.shape[1], z.shape[2], math.ceil(z.shape[2]/self.n_intra), z.shape[-1], device=device)  # padded: [b, c, num_period, num_dynamic, period]
        flag = 0
        for u in range(0, num_period, int(self.n_intra)):  # u: [0, num_period-1]
            start_idx1 = u
            end_idx1 = z.shape[2]
            start_idx2 = 0
            end_idx2 = z.shape[2] - u
            padded_intra[:, :, start_idx1:end_idx1, flag, :] = z[:, :, start_idx2:end_idx2, :]
            flag += 1
        z_intra = padded_intra.view(-1, padded_intra.shape[-3], padded_intra.shape[-2] * padded_intra.shape[-1])     # [b*c, num_period, num_dynamic*period]
        # model
        z_intra = self.backbone(z_intra, 'intra')                                        # z: [b*c, num_period, d_model]
        z_intra = z_intra.reshape(b, c, -1)                                        # z: [b, c, num_period*d_model]
        
        ######################## fine-scale
        padded = torch.zeros(z.shape[0], z.shape[1], z.shape[2], math.ceil(z.shape[-1]/self.n), z.shape[-1], device=device)  # padded: [b, c, num_period, num_dynamic, period]
        flagg = 0
        for uu in range(0, period, int(self.n)):  # u: [0, num_dynamic-1]
            start_idx1 = uu
            end_idx1 = z.shape[-1]
            start_idx2 = 0
            end_idx2 = z.shape[-1] - uu
            padded[:, :, :, flagg, start_idx1:end_idx1] = z[:, :, :, start_idx2:end_idx2]
            flagg += 1
        z_inter = padded.view(-1, padded.shape[-2], padded.shape[-1])        # [b*c*num_period, num_dynamic, period]
        # model
        z_inter = self.backbone(z_inter, 'inter')                                  # z: [b*c*num_period, period, d_model]    
        z_inter = z_inter.reshape(b, c, -1)                                  # z: [b, c, num_period*period*d_model]
        
        z = torch.cat([z_intra, z_inter, z_coarse], dim=-1)        
        z = self.head(z)                                                     # z: [bs x channel x target_window]
        # denorm
        if self.revin:
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z



class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, coarse_len, c_in, context_window, n, n_intra, num_period, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        # Input encoding
        self.q_len = context_window
        # intra
        self.W_P_intra = nn.Linear(math.ceil(num_period/n_intra)*self.q_len, d_model)  # num_dynamic*period -> d
        self.W_pos_intra = positional_encoding(pe, learn_pe, num_period, d_model)
        # inter
        self.W_P = nn.Linear(math.ceil(self.q_len/n), d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.W_pos = positional_encoding(pe, learn_pe, self.q_len, d_model)
        # coarse
        self.W_P_coarse = nn.Linear(coarse_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.W_pos_coarse = positional_encoding(pe, learn_pe, c_in, d_model)
        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        # Encoder
        self.encoder = TSTEncoder(self.q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)
        self.encoder_intra = TSTEncoder(self.q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)
        self.encoder_coarse = TSTEncoder(self.q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

    def forward(self, x, mode) -> Tensor:                                           # x: [b*c, num_period, num_dynamic*period]
        if mode=='intra':
            x = self.W_P_intra(x)                                                    # x: [b*c, num_period, d_model]
            u = self.dropout(x + self.W_pos_intra)                                   # u: [b*c, num_period, d_model]
            z = self.encoder_intra(u)                                                # z: [b*c, num_period, d_model]
        elif mode=='inter':
            x = x.permute(0,2,1)                                                     # x: [bcb, boundary_num_dynamic, boundary_period]->[bcb x boundary_period x boundary_num_dynamic], bcb = b*c*boundary_num_period, boundary_num_dynamic = boundary_period
            x = self.W_P(x)                                                          # x: [bcb x boundary_period x d_model]
            u = self.dropout(x + self.W_pos)                                         # u: [bcb x boundary_period x d_model]
            z = self.encoder(u)                                                      # z: [bcb x boundary_period x d_model]
        elif mode=='coarse':
            x = self.W_P_coarse(x)                                                    # x: [b, c, d_model]
            u = self.dropout(x + self.W_pos_coarse)                                   # u: [b, c, d_model]
            z = self.encoder_coarse(u)                                                # z: [b, c, d_model]
        return z



# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        # print('src:',output)
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                # print('mod:',mod)
                # print('output:',output)
                # print('scores:', scores)
            # print('res_attention', output)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            # print('no_res_attention', output)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        # print('src2:',src)
        if self.pre_norm:
            # print('11')
            src = self.norm_attn(src)
            # print(src)
        ## Multi-Head attention
        if self.res_attention:
            # print('22')
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            # print(src2)
        else:
            # print('33')
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q
        # print('mask:', key_padding_mask)
        # print('QQ:',Q)
        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]
        # print('q_s:',q_s)
        # print('k_s:',k_s)
        # print('v_s', v_s)
        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            # print('output:', output)
            # print('attn_weights:', attn_weights)
            # print('attn_scores:', attn_scores)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''
        # print('q_shape:',q.shape)
        # print('k_shape:', k.shape)
        # print('v_shape:', v.shape)
        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]
        # print('attn_scores:',attn_scores)
        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        # print('attn_scores2:', attn_scores)
        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
        # print('attn_scores3:', attn_scores)
        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        # print('attn_weights:', attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

