from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from ldm.modules.diffusionmodules.util import checkpoint


try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0 , show_attnmap = False , ex_attn = False):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.show = show_attnmap
        self.ex_attn = ex_attn

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None ,timestep = None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        if self.show:
        # 判断是否可视化attentionmap
            ## batchsize 
            
            attn_map = self.extra_atttentionmap(q,k,b,False)
            # print("attn_map size",attn_map.shape)
            # 可视化每个头的attentionmap的结果
            self.visualize_cross_attention(attn_map,b)


        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        # print("out size",out.shape)
        # cross attention 的 结果处理

        # 提取attn的结果
        if self.ex_attn:
            attn_map = self.extra_atttentionmap(q,k,b,False)
            return self.to_out(out),attn_map
        
        return self.to_out(out)
    

    ###  我自己写的提取attention map 函数：
    def extra_atttentionmap(self,query,key,bs,show=True,scale = None,attn_bias = None):
        # print("quary size",query.shape)
        # print("key size",key.shape)
        scale = 1 / query.shape[-1] ** 0.5
        query = query * scale
        attn = torch.matmul(query,key.transpose(-2,-1))
        # attn = torch.matmul(query.transpose(-2, -1), key)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = F.softmax(attn, dim=-1)
        if show:
            c ,h ,w = attn.shape
            attn = attn[0:bs*self.heads]
            attn = attn.reshape(bs,self.heads,h,w)
            return attn
        else:
            return attn
    
    def visualize_self_attention(self,multihead_attn_map,threshold=0.5, save_dir="/home/wzb/workspace/ControlNet-main/attrntion_map_result/mask_to_img/t=0"):
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2
        batch_size, num_heads, num_queries, num_keys = multihead_attn_map.shape
        for i in range(batch_size):
            # attention_mask = multihead_attn_map[i].cpu().numpy()
            attention_mask = multihead_attn_map[i].cpu().numpy()  # 对每个批次的所有头求均值
            # print(attention_mask.shape)
            attention_mask = attention_mask.reshape(attention_mask.shape[0],int(np.sqrt(attention_mask.shape[1])),-1)
            attention_mask = np.sum(attention_mask,axis=0)
            mask = cv2.resize(attention_mask, (512,512))
            min_val = np.min(mask)
            max_val = np.max(mask)
            normed_mask = (mask - min_val) / (max_val - min_val)
            # normed_mask = mask / mask.max()
            normed_mask = (normed_mask * 255).astype('uint8')
            # normed_mask = (normed_mask * 255).astype('uint8')
            plt.imshow(normed_mask, alpha=None, interpolation='nearest', cmap="jet")
            # fig, ax = plt.subplots(figsize=(8, 8), dpi=100)  # 设置图形大小为512x512像素
            # binary_map = (average_map >= threshold).astype(float)  # 将平均注意力图二值化
            plt.savefig(f"{save_dir}/binary_attention_map_batch_{i+1}.png")

    
            plt.close()

    def visualize_cross_attention(self,multihead_attn_map, b,root="/home/wzb/workspace/ControlNet-main/SD_cross_attention_map/"):
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2
        import os
        if not os.path.exists(root):
            os.makedirs(root)
        sliced_tensor = np.split(multihead_attn_map, b, axis=0)
        for i,attention_mask in  enumerate(sliced_tensor):
                # attention_mask = multihead_attn_map[i].cpu().numpy()
                attention_mask = attention_mask.cpu().detach().numpy()  # 对每个批次的所有头求均值
                # print(attention_mask.shape)
                attention_mask = np.mean(attention_mask,axis=0)
                attention_mask = attention_mask.reshape(int(np.sqrt(attention_mask.shape[0])),-1,77)
                attention_mask = attention_mask[:,:,0]

                # mask = cv2.resize(attention_mask,(512,512))
                mask = attention_mask
                min_val = np.min(mask)
                max_val = np.max(mask)
                normed_mask = (mask - min_val) / (max_val - min_val)
                normed_mask = (normed_mask * 255).astype('uint8')
                plt.imshow(normed_mask, alpha=None, interpolation='nearest', cmap="jet")
                # fig, ax = plt.subplots(figsize=(8, 8), dpi=100)  # 设置图形大小为512x512像素
                # binary_map = (average_map >= threshold).astype(float)  # 将平均注意力图二值化

                plt.savefig(f"{root}/binary_attention_map_batch_{i+1}.png")

                plt.close()
    

class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False,show_attnmap = False,ex_crossattn = False,ex_selfattn = False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        assert not( ex_crossattn and  ex_selfattn) , "only extract one"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None ,ex_attn = ex_selfattn)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout, show_attnmap = show_attnmap , ex_attn = ex_crossattn)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.excrossattn = ex_crossattn
        self.exselfattn = ex_selfattn

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        
        if not self.excrossattn and not self.exselfattn:
            x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
            x = self.attn2(self.norm2(x), context=context) + x
            
        else:
            if self.exselfattn:
                a , attn = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None)
                x = a + x
                x = self.attn2(self.norm2(x), context=context) + x
            else:
                x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
                a , attn = self.attn2(self.norm2(x), context=context)
                x = a + x
        x = self.ff(self.norm3(x)) + x
        if self.excrossattn or self.exselfattn:
            return x , attn
        return x

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True,show_attnmap = False , ex_crossattn = False,ex_selfattn = False):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint , show_attnmap = show_attnmap , ex_crossattn = ex_crossattn , ex_selfattn= ex_selfattn)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear
        self.ex_attn = False
        if ex_crossattn or ex_selfattn:
            self.ex_attn = True
    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            ## 判断是否需要返回attn的值
            if self.ex_attn:
                x , att  = block(x, context=context[i])
            else:
                x = block(x, context = context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        if self.ex_attn:
            return x + x_in , att
        return x + x_in

