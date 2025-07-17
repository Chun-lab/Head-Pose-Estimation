# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# import os
# import math
# import torch
# import torch.nn.functional as F
# from einops import rearrange, repeat
# from torch import nn
# from timm.layers.weight_init import trunc_normal_
# from ViT_model import VisionTransformer
# from utils import compute_rotation_matrix_from_ortho6d
# import matplotlib.pyplot as plt
# import seaborn as sns
# MIN_NUM_PATCHES = 16
# BN_MOMENTUM = 0.1
#
# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#
#     def forward(self, x, **kwargs):
#         return self.fn(x, **kwargs) + x
#
#
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn, fusion_factor=1):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim * fusion_factor)
#         self.fn = fn
#
#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)
#
#
# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.):
#         super().__init__()
#
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# class Attention(nn.Module):
#     def __init__(self, dim, heads=8, dropout=0., num_ori_tokens=None, scale_with_head=False, show_attns=False,n_dep=0):
#         super().__init__()
#         self.heads = heads
#         self.scale = (dim // heads) ** -0.5 if scale_with_head else dim ** -0.5
#
#         self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
#         self.to_out = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.Dropout(dropout)
#         )
#         self.show_attns = show_attns
#         self.num_ori_tokens = num_ori_tokens
#         self.n_dep = n_dep
#
#     def plot_attention(self, attn, type="single head", head_index=5):
#         if not os.path.exists('output/vis'):
#             os.makedirs('output/vis')
#         if type == "single head":
#             values = attn[0,head_index,0:self.num_ori_tokens,0:self.num_ori_tokens].detach().cpu()
#         else: # all heads
#             values = torch.sum(attn,dim=1)
#             values = values[0, 0:self.num_ori_tokens, 0:self.num_ori_tokens].detach().cpu()
#
#         fig = plt.figure()
#         sns.heatmap(values, cmap='plasma')
#         fig.savefig(f"./output/vis/attention interaction in layer {self.n_dep+1}.png", bbox_inches='tight')
#         plt.show()
#
#     def forward(self, x, mask=None):
#         b, n, _, h = *x.shape, self.heads
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
#
#         dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
#         mask_value = -torch.finfo(dots.dtype).max
#
#         if mask is not None:
#             mask = F.pad(mask.flatten(1), (1, 0), value=True)
#             assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
#             mask = mask[:, None, :] * mask[:, :, None]
#             dots.masked_fill_(~mask, mask_value)
#             del mask
#
#         attn = dots.softmax(dim=-1)
#
#         if self.show_attns == True:
#             self.plot_attention(attn)
#
#         out = torch.einsum('bhij,bhjd->bhid', attn, v)
#
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)
#         return out
#
#
# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, mlp_dim, dropout, num_ori_tokens=None,
#                  all_attn=False, scale_with_head=False, show_attns=False):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         self.all_attn = all_attn
#         self.num_ori_tokens = num_ori_tokens
#         for n_dep in range(depth):
#             self.layers.append(nn.ModuleList([
#                 Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout, num_ori_tokens=num_ori_tokens,
#                                                 scale_with_head=scale_with_head,show_attns=show_attns,n_dep=n_dep))),
#                 Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
#             ]))
#
#     def forward(self, x, mask=None, pos=None):
#         for idx, (attn, ff) in enumerate(self.layers):
#             if idx > 0 and self.all_attn:
#                 x[:, self.num_ori_tokens:] += pos
#             # print(x.shape)
#             # [64,207,128]
#             x = attn(x, mask=mask)
#             x = ff(x)
#             # print(x.shape)
#         return x
#
#
# class Orientation_Blocks(nn.Module):
#     """
#     feature extractor (ViT) -> Orientation_Blocks -> outputs in all regions
#     """
#     def __init__(self, *, num_ori_tokens, dim, depth, heads, mlp_dim,
#                  dropout=0., emb_dropout=0., pos_embedding_type="learnable",
#                  ViT_feature_dim, ViT_feature_num, w, h, inference_view=False):
#         """
#         inference_view: In inference stage, for a single image input, show the ori_tokens similarity matrix
#         and the attention interactions of the learned ori_tokens in each Transformer layer.
#         """
#         super().__init__()
#         patch_dim = ViT_feature_dim
#         self.inplanes = 64
#         self.num_ori_tokens = num_ori_tokens
#         self.num_patches = ViT_feature_num
#         self.pos_embedding_type = pos_embedding_type
#         self.all_attn = (self.pos_embedding_type == "sine-full")
#
#         self.ori_tokens = nn.Parameter(torch.zeros(1, self.num_ori_tokens, dim))
#
#         self._make_position_embedding(w, h, dim, pos_embedding_type)
#
#         self.patch_to_embedding = nn.Linear(patch_dim, dim)
#         self.dropout = nn.Dropout(emb_dropout)
#         self.inference_view = inference_view
#
#         self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, num_ori_tokens=num_ori_tokens,
#                                        all_attn=self.all_attn, show_attns=self.inference_view)
#
#         self.to_ori_token = nn.Identity()
#
#         self.to_dir_6_d = nn.Sequential(
#             nn.Linear(dim, 6)
#         )
#
#     def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
#         '''
#         d_model: embedding size in transformer encoder
#         '''
#         assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
#         if pe_type == 'none':
#             self.pos_embedding = None
#             print("==> Without any PositionEmbedding~")
#         else:
#             with torch.no_grad():
#                 self.pe_h = h
#                 self.pe_w = w
#                 length = self.pe_h * self.pe_w
#             if pe_type == 'learnable':
#                 self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_ori_tokens, d_model))
#                 trunc_normal_(self.pos_embedding, std=.02)
#                 print("==> Add Learnable PositionEmbedding~")
#             else:
#                 self.pos_embedding = nn.Parameter(
#                     self._make_sine_position_embedding(d_model),
#                     requires_grad=False)
#                 print("==> Add Sine PositionEmbedding~")
#
#     def _make_sine_position_embedding(self, d_model, temperature=10000,
#                                       scale=2 * math.pi):
#         h, w = self.pe_h, self.pe_w
#         area = torch.ones(1, h, w)  # [b, h, w]
#         y_embed = area.cumsum(1, dtype=torch.float32)
#         x_embed = area.cumsum(2, dtype=torch.float32)
#
#         one_direction_feats = d_model // 2
#
#         eps = 1e-6
#         y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
#         x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale
#
#         dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
#         dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)
#
#         pos_x = x_embed[:, :, :, None] / dim_t
#         pos_y = y_embed[:, :, :, None] / dim_t
#         pos_x = torch.stack(
#             (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
#         pos_y = torch.stack(
#             (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
#         pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
#         pos = pos.flatten(2).permute(0, 2, 1)
#         return pos
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def _init_weights(self, m):
#         print("Initialization...")
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def plot_sim_matrix(self, A, type="cos-similarity"):
#         """
#         visualize the similarity matrix of orientation tokens
#         type: "softmax/cos-similarity"
#         """
#         assert type == "softmax" or type == "cos-similarity", "please use correct type. (softmax/cos-similarity)"
#
#         if not os.path.exists('output/vis'):
#             os.makedirs('output/vis')
#
#         if type=="softmax":
#             in_pro = A.mm(A.T) / math.sqrt(A.shape[1])
#             prob = F.softmax(in_pro, dim=0)
#             fig = plt.figure()
#             sns.heatmap(prob, cmap='plasma')
#             fig.savefig("./output/vis/softmax_similarity_matrix_of_ori_tokens.png", bbox_inches='tight')
#             plt.show()
#         elif type=="cos-similarity":
#             a = (A / torch.norm(A, dim=-1, keepdim=True) )[0,...]
#             similarity = torch.mm(a, a.T)
#             fig = plt.figure()
#             sns.heatmap(similarity, cmap='plasma') # options: YlGnBu plasma
#             fig.savefig("./output/vis/cos_similarity_matrix_of_ori_tokens.png", bbox_inches='tight')
#             plt.show()
#
#     def forward(self, features, mask=None):
#         """
#         feature extractor (ViT) -> add ori_tokens -> Orientation_Blocks -> outputs in all regions
#         """
#         # show ori_tokens similarity matrix
#         if self.inference_view == True:
#             self.plot_sim_matrix(self.ori_tokens.cpu())
#
#         # transformer features
#         # features[64,196,768]
#         x = self.patch_to_embedding(features) # shape [batch_size, channel=197, dim=192]
#         # [64, 196, 128]
#         b, n, _ = x.shape
#
#         # add learnable orientation tokens
#         ori_tokens = repeat(self.ori_tokens, '() n d -> b n d', b=b)
#
#         # add pos_embedding
#         if self.pos_embedding_type in ["sine", "sine-full"]:
#             x += self.pos_embedding[:, :n]
#             x = torch.cat((ori_tokens, x), dim=1)
#         elif self.pos_embedding_type == "learnable":
#             x = torch.cat((ori_tokens, x), dim=1)
#             x += self.pos_embedding[:, :(n + self.num_ori_tokens)]
#
#         x = self.dropout(x)
#
#         # feature extractor (ViT) -> Orientation_Blocks
#         x = self.transformer(x, mask, self.pos_embedding)
#
#         # Orientation_Blocks -> outputs in all regions
#         dir_tokens = self.to_ori_token(x[:, 0:self.num_ori_tokens])
#         dir_6_d = self.to_dir_6_d(dir_tokens)
#
#         # convert to rotation matrices
#         batch_size, num_ori_tokens, d = dir_6_d.size()
#         x_reshaped = dir_6_d.view(-1, d)
#         ori_9_d = compute_rotation_matrix_from_ortho6d(x_reshaped)
#
#         # reshape to [batch_size, num_ori_tokens, 3d, 3d]
#         ori_9_d = ori_9_d.view(batch_size, num_ori_tokens, 3,3)
#
#         return ori_9_d
#
#
# class TokenHPE(nn.Module):
#
#     def __init__(self, num_ori_tokens=11,
#                  depth=12, heads=12, embedding='sine-full', ViT_weights='',
#                  dim=128, mlp_ratio=3, inference_view=False
#                  ):
#         super(TokenHPE, self).__init__()
#
#         # Feature extractor (ViT)
#         # VisionTransformer implemented by rwightman:
#         # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
#         # use vit_base_patch16_224_in21k
#         self.feature_extractor = VisionTransformer(
#                               img_size=224,
#                               patch_size=16,
#                               embed_dim=768,
#                               depth=12,
#                               num_heads=12,
#                               representation_size=None,
#                               mlp_head=False,
#                               )
#
#         # whether to use intermediate weights
#         if ViT_weights != "":
#             assert os.path.exists(ViT_weights), "weights file: '{}' not exist.".format(ViT_weights)
#             weights_dict = torch.load(ViT_weights, map_location="cuda")
#             # delete cls head weights
#             for k in list(weights_dict.keys()):
#                 if "head" in k:
#                     del weights_dict[k]
#             print("use pretrained feature extractor (ViT) weights!")
#             print(self.feature_extractor.load_state_dict(weights_dict, strict=False))
#
#         # Transformer blocks with orientation tokens
#         self.Ori_blocks = Orientation_Blocks(
#                                          num_ori_tokens=num_ori_tokens,
#                                          dim=dim,
#                                          ViT_feature_dim=768,
#                                          ViT_feature_num=197,
#                                          w=14,
#                                          h=14,
#                                          depth=depth,
#                                          heads=heads,
#                                          mlp_dim=dim * mlp_ratio,
#                                          pos_embedding_type=embedding,
#                                          inference_view=inference_view
#                                          )
#
#         self.mlp_head = nn.Sequential(
#             nn.Linear(num_ori_tokens*9, num_ori_tokens*27),
#             nn.Tanh(),
#             nn.Linear(num_ori_tokens*27, 6)
#         )
#
#     def forward(self, x):
#         """
#         TokenHPE pipeline
#         feature extractor (ViT) -> Orientation_Blocks -> outputs in all regions
#         -> MLP head -> prediction: [pred, ori_9_d]
#         """
#         # feature extractor (ViT)
#         x = self.feature_extractor(x) # outputs: [batch_size, channel=197, dim = 768]
#
#         # Orientation_Blocks
#         ori_9_d = self.Ori_blocks(x) # [batch_size, num_ori_tokens, 3d, 3d]
#
#         # feed to mlp head
#         x = rearrange(ori_9_d, 'batch oris d_1 d_2-> batch (oris d_1 d_2)')
#
#         x = self.mlp_head(x)
#
#         pred = compute_rotation_matrix_from_ortho6d(x)
#
#         return pred, ori_9_d


# zuihou
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import math
import torch
from timm.layers import DropPath, Mlp, to_2tuple
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from timm.layers.weight_init import trunc_normal_
from ViT_model import VisionTransformer
from utils import compute_rotation_matrix_from_ortho6d
import matplotlib.pyplot as plt
import seaborn as sns

MIN_NUM_PATCHES = 16
BN_MOMENTUM = 0.1


class CNN(nn.Module):
    def __init__(self, inplanes=64, embed_dim=128):
        super().__init__()

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SELayer(nn.Module):
    def __init__(self, channel, r=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()  # [64,207,128]
        y = self.avg_pool(x).view(b, c)  # batch, channel
        # [64,207]
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SynthAttention(nn.Module):
    def __init__(self, in_dims, out_dims, reduce_factor=1):  # L, emb
        super().__init__()

        reduced_dims = out_dims // reduce_factor
        self.dense = nn.Linear(in_dims, reduced_dims)
        self.reduce_factor = reduce_factor
        self.se = SELayer(out_dims, r=4)
        self.value_reduce = nn.Linear(out_dims, reduced_dims)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        energy = self.dense(x)  # b, channel, reduced_dim
        energy = self.se(energy)
        attention = self.softmax(energy)

        value = x
        if self.reduce_factor > 1:
            value = self.value_reduce(value.transpose(1, 2)).transpose(1, 2)  # b, reduced_dim, t

        out = torch.bmm(attention, value)
        return out


# class Attention(nn.Module):
#     def __init__(self, dim, heads=8, dropout=0., num_ori_tokens=None, scale_with_head=False, show_attns=False, n_dep=0):
#         super().__init__()
#         self.heads = heads
#         self.scale = (dim // heads) ** -0.5 if scale_with_head else dim ** -0.5
#
#         self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
#         self.to_out = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.Dropout(dropout)
#         )
#         self.show_attns = show_attns
#         self.num_ori_tokens = num_ori_tokens
#         self.n_dep = n_dep
#         self.gap = nn.AdaptiveAvgPool1d(1)
#
#     def plot_attention(self, attn, type="single head", head_index=5):
#         if not os.path.exists('output/vis'):
#             os.makedirs('output/vis')
#         if type == "single head":
#             values = attn[0, head_index, 0:self.num_ori_tokens, 0:self.num_ori_tokens].detach().cpu()
#         else:  # all heads
#             values = torch.sum(attn, dim=1)
#             values = values[0, 0:self.num_ori_tokens, 0:self.num_ori_tokens].detach().cpu()
#
#         fig = plt.figure()
#         sns.heatmap(values, cmap='plasma')
#         fig.savefig(f"./output/vis/attention interaction in layer {self.n_dep + 1}.png", bbox_inches='tight')
#         plt.show()
#
#     def forward(self, x, mask=None):
#         b, n, _, h = *x.shape, self.heads
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
#
#         dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
#         mask_value = -torch.finfo(dots.dtype).max
#
#         if mask is not None:
#             mask = F.pad(mask.flatten(1), (1, 0), value=True)
#             assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
#             mask = mask[:, None, :] * mask[:, :, None]
#             dots.masked_fill_(~mask, mask_value)
#             del mask
#
#         attn = dots.softmax(dim=-1)
#
#         if self.show_attns == True:
#             self.plot_attention(attn)
#
#         out = torch.einsum('bhij,bhjd->bhid', attn, v)
#
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)
#
#         return out
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0., num_ori_tokens=None, scale_with_head=False, show_attns=False, n_dep=0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5 if scale_with_head else dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.show_attns = show_attns
        self.num_ori_tokens = num_ori_tokens
        self.n_dep = n_dep
        self.gap = nn.AdaptiveAvgPool1d(1)

    def plot_attention(self, attn, type="single head", head_index=5):
        if not os.path.exists('output/vis'):
            os.makedirs('output/vis')
        if type == "single head":
            values = attn[0, head_index, 0:self.num_ori_tokens, 0:self.num_ori_tokens].detach().cpu()
        else:  # all heads
            values = torch.sum(attn, dim=1)
            values = values[0, 0:self.num_ori_tokens, 0:self.num_ori_tokens].detach().cpu()

        fig = plt.figure()
        sns.heatmap(values, cmap='plasma')
        fig.savefig(f"./output/vis/attention interaction in layer {self.n_dep + 1}.png", bbox_inches='tight')
        plt.show()

    def forward(self, x, y, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        qkv1 = self.to_qkv(y).chunk(3, dim=-1)
        q1, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        q, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv1)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        if self.show_attns == True:
            self.plot_attention(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class SynthMix(nn.Module):
    def __init__(self, temporal_dim=207, channel_dim=128, proj_drop=0.1, reduce_token=1, reduce_channel=1):  # L, emb
        super().__init__()
        # generate temporal matrix
        self.synth_token = SynthAttention(channel_dim, temporal_dim,
                                          reduce_token)  # reduce factor for window # l, emb -> l, l * l, emb
        # generate spatial matrix
        self.synth_channel = SynthAttention(temporal_dim, channel_dim,
                                            reduce_channel)  # reduce factor for embedding # emb, l -> emb, emb//r * emb//r, l

        self.reweight = Mlp(temporal_dim, temporal_dim // 4, temporal_dim * 2)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.proj = nn.Linear(channel_dim, channel_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        b, _, _ = x.shape
        t = self.synth_token(x)  # b, t, c
        c = y  # b, c, t

        # re-weight
        t = t.transpose(1, 2)
        c = c.transpose(1, 2)
        a = self.gap((t + c).transpose(1, 2)).squeeze(-1)  # shape: batch, emb
        a = self.reweight(a).reshape(b, 207, 2).permute(2, 0, 1).softmax(dim=0)  # 2, batch, channel
        s = torch.einsum("nble, nbe -> nble", [torch.stack([t, c], 0), a])
        x = torch.sum(s, dim=0)
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        dim1 = dim
        dim = dim // 2

        self.dwconv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)

        self.dwconv3 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv4 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)

        self.dwconv5 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv6 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)

        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(dim1)

        self.act2 = nn.GELU()
        self.bn2 = nn.BatchNorm2d(dim1)

        self.act3 = nn.GELU()
        self.bn3 = nn.BatchNorm2d(dim1)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()

        x11, x12 = x1[:, :C // 2, :, :], x1[:, C // 2:, :, :]
        x11 = self.dwconv1(x11)  # BxCxHxW
        x12 = self.dwconv2(x12)
        x1 = torch.cat([x11, x12], dim=1)
        x1 = self.act1(self.bn1(x1)).flatten(2).transpose(1, 2)

        x21, x22 = x2[:, :C // 2, :, :], x2[:, C // 2:, :, :]
        x21 = self.dwconv3(x21)
        x22 = self.dwconv4(x22)
        x2 = torch.cat([x21, x22], dim=1)
        x2 = self.act2(self.bn2(x2)).flatten(2).transpose(1, 2)

        x31, x32 = x3[:, :C // 2, :, :], x3[:, C // 2:, :, :]
        x31 = self.dwconv5(x31)
        x32 = self.dwconv6(x32)
        x3 = torch.cat([x31, x32], dim=1)
        x3 = self.act3(self.bn3(x3)).flatten(2).transpose(1, 2)

        x = torch.cat([x1, x2, x3], dim=1)
        return x


class MRFP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = MultiDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




def activation_fn(x):
    return torch.relu(x)


class Back(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.reweight = Mlp(207, 207 // 4, 207 * 2)
        self.proj = nn.Linear(128, 128)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x, y):
        b, _, _ = x.shape
        x1_new = x.transpose(1, 2)
        x2_new = y.transpose(1, 2)
        a = self.gap((x1_new + x2_new).transpose(1, 2)).squeeze(-1)
        a = self.reweight(a).reshape(b, 207, 2).permute(2, 0, 1).softmax(dim=0)
        s = torch.einsum("nble, nbe -> nble", [torch.stack([x1_new, x2_new], 0), a])
        y = torch.sum(s, dim=0)
        y = y.transpose(1, 2)
        y = self.proj(y)
        y = self.proj_drop(y)

        return y


class SynthEncoderLayer(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        self.ff = Mlp(in_features=d_model, hidden_features=d_model * expansion_factor, out_features=d_model,
                      drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, y):
        x = self.norm1(x + y)
        x = self.norm2(x + self.ff(x))
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim * fusion_factor)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, num_ori_tokens=None,
                 all_attn=False, scale_with_head=False, show_attns=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.all_attn = all_attn
        self.norm = nn.LayerNorm(dim)
        self.num_ori_tokens = num_ori_tokens
        for n_dep in range(depth):
            self.layers.append(nn.ModuleList([
                # Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout, num_ori_tokens=num_ori_tokens,
                #                                 scale_with_head=scale_with_head, show_attns=show_attns, n_dep=n_dep))),
                Attention(dim, heads=heads, dropout=dropout, num_ori_tokens=num_ori_tokens,
                          scale_with_head=scale_with_head, show_attns=show_attns, n_dep=n_dep),
                # PreNorm(dim, Attention(dim, heads=heads, dropout=dropout, num_ori_tokens=num_ori_tokens,
                #                                                        scale_with_head=scale_with_head, show_attns=show_attns, n_dep=n_dep)),
                SynthMix(temporal_dim=207, channel_dim=128, proj_drop=0.1, reduce_token=1, reduce_channel=1),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                Back(),
                SynthEncoderLayer(d_model=128, expansion_factor=1, dropout=0.1)
            ]))

    def forward(self, x, y, mask=None, pos=None):
        for idx, (attn, syn, ff, back, encoder) in enumerate(self.layers):
            if idx > 0 and self.all_attn:
                x[:, self.num_ori_tokens:] += pos
            x1 = attn(x, y) + x #[1,207,128]
            y = syn(y, x1)
            x1 = ff(x1)
            y = back(x1, y)
            y = encoder(x, y)

        return x1, y


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Orientation_Blocks(nn.Module):
    """
    feature extractor (ViT) -> Orientation_Blocks -> outputs in all regions
    """

    def __init__(self, *, num_ori_tokens, dim, depth, heads, mlp_dim,
                 dropout=0., emb_dropout=0., pos_embedding_type="learnable",
                 ViT_feature_dim, ViT_feature_num, w, h, inference_view=False):
        """
        inference_view: In inference stage, for a single image input, show the ori_tokens similarity matrix
        and the attention interactions of the learned ori_tokens in each Transformer layer.
        """
        super().__init__()
        patch_dim = ViT_feature_dim
        self.inplanes = 64
        self.num_ori_tokens = num_ori_tokens
        self.num_patches = ViT_feature_num
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")

        self.ori_tokens = nn.Parameter(torch.zeros(1, self.num_ori_tokens, dim))

        self._make_position_embedding(w, h, dim, pos_embedding_type)

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.inference_view = inference_view

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, num_ori_tokens=num_ori_tokens,
                                       all_attn=self.all_attn, show_attns=self.inference_view)

        self.to_ori_token = nn.Identity()

        self.to_dir_6_d = nn.Sequential(
            nn.Linear(dim, 6)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_ori_tokens, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        print("Initialization...")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def plot_sim_matrix(self, A, type="cos-similarity"):
        """
        visualize the similarity matrix of orientation tokens
        type: "softmax/cos-similarity"
        """
        assert type == "softmax" or type == "cos-similarity", "please use correct type. (softmax/cos-similarity)"

        if not os.path.exists('output/vis'):
            os.makedirs('output/vis')

        if type == "softmax":
            in_pro = A.mm(A.T) / math.sqrt(A.shape[1])
            prob = F.softmax(in_pro, dim=0)
            fig = plt.figure()
            sns.heatmap(prob, cmap='plasma')
            fig.savefig("./output/vis/softmax_similarity_matrix_of_ori_tokens.png", bbox_inches='tight')
            plt.show()
        elif type == "cos-similarity":
            a = (A / torch.norm(A, dim=-1, keepdim=True))[0, ...]
            similarity = torch.mm(a, a.T)
            fig = plt.figure()
            sns.heatmap(similarity, cmap='plasma')  # options: YlGnBu plasma
            fig.savefig("./output/vis/cos_similarity_matrix_of_ori_tokens.png", bbox_inches='tight')
            plt.show()

    def forward(self, features, features1, mask=None):
        """
        feature extractor (ViT) -> add ori_tokens -> Orientation_Blocks -> outputs in all regions
        """
        # show ori_tokens similarity matrix
        if self.inference_view == True:
            self.plot_sim_matrix(self.ori_tokens.cpu())

        # transformer features
        # features[64,196,768]
        x = self.patch_to_embedding(features)  # shape [batch_size, channel=197, dim=192]
        y = self.patch_to_embedding(features1)
        # [64, 196, 128]
        b, n, _ = x.shape

        # add learnable orientation tokens
        ori_tokens = repeat(self.ori_tokens, '() n d -> b n d', b=b)

        # add pos_embedding
        if self.pos_embedding_type in ["sine", "sine-full"]:
            x += self.pos_embedding[:, :n]
            x = torch.cat((ori_tokens, x), dim=1)
            y += self.pos_embedding[:, :n]
            y = torch.cat((ori_tokens, x), dim=1)
        elif self.pos_embedding_type == "learnable":
            x = torch.cat((ori_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + self.num_ori_tokens)]
            y = torch.cat((ori_tokens, x), dim=1)
            y += self.pos_embedding[:, :(n + self.num_ori_tokens)]

        x = self.dropout(x)
        y = self.dropout(x)

        # feature extractor (ViT) -> Orientation_Blocks
        x, y = self.transformer(x, y, mask, self.pos_embedding)

        # Orientation_Blocks -> outputs in all regions
        # dir_tokens = self.to_ori_token(x[:, 0:self.num_ori_tokens])
        # dir_6_d = self.to_dir_6_d(dir_tokens)
        dir_tokens = self.to_ori_token(y[:, 0:self.num_ori_tokens]) # change
        dir_6_d = self.to_dir_6_d(dir_tokens)

        # convert to rotation matrices
        batch_size, num_ori_tokens, d = dir_6_d.size()
        x_reshaped = dir_6_d.view(-1, d)
        ori_9_d = compute_rotation_matrix_from_ortho6d(x_reshaped)

        # reshape to [batch_size, num_ori_tokens, 3d, 3d]
        ori_9_d = ori_9_d.view(batch_size, num_ori_tokens, 3, 3)

        return ori_9_d


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten=True, bias=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, H, W


class TokenHPE(nn.Module):

    def __init__(self, num_ori_tokens=11,
                 depth=12, heads=12, embedding='sine-full', ViT_weights='',
                 dim=128, mlp_ratio=3, inference_view=False
                 ):
        super(TokenHPE, self).__init__()

        # Feature extractor (ViT)
        # VisionTransformer implemented by rwightman:
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # use vit_base_patch16_224_in21k
        self.feature_extractor = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            representation_size=None,
            mlp_head=False,
        )

        # whether to use intermediate weights
        if ViT_weights != "":
            assert os.path.exists(ViT_weights), "weights file: '{}' not exist.".format(ViT_weights)
            weights_dict = torch.load(ViT_weights, map_location="cuda")
            # delete cls head weights
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print("use pretrained feature extractor (ViT) weights!")
            print(self.feature_extractor.load_state_dict(weights_dict, strict=False))

        # Transformer blocks with orientation tokens
        self.Ori_blocks = Orientation_Blocks(
            num_ori_tokens=num_ori_tokens,
            dim=dim,
            ViT_feature_dim=768,
            ViT_feature_num=197,
            w=14,
            h=14,
            depth=depth,
            heads=heads,
            mlp_dim=dim * mlp_ratio,
            pos_embedding_type=embedding,
            inference_view=inference_view
        )
        self.mrfp = MRFP(768, hidden_features=int(768 * 0.5))
        self.mlp_head = nn.Sequential(
            nn.Linear(num_ori_tokens * 9, num_ori_tokens * 27),
            nn.Tanh(),
            nn.Linear(num_ori_tokens * 27, 6)
        )
        self.spm = CNN(inplanes=64,
                       embed_dim=768)
        self.level_embed = nn.Parameter(torch.zeros(3, 768))
    # end forward
    # def forward(self, x):
    #     """
    #     TokenHPE pipeline
    #     feature extractor (ViT) -> Orientation_Blocks -> outputs in all regions
    #     -> MLP head -> prediction: [pred, ori_9_d]
    #     """
    #     # feature extractor (ViT)
    #     c1, c2, c3, c4 = self.spm(x)
    #     x = self.feature_extractor(x)  # outputs: [batch_size, channel=197, dim = 768]
    #
    #     c = torch.cat([c2, c3, c4], dim=1)
    #     c = self.mrfp(c, 14, 14)
    #     H = 14
    #     W = 14
    #     c_select1, c_select2, c_select3 = c[:, :H * W * 4, :], c[:, H * W * 4:H * W * 4 + H * W, :], c[:,
    #                                                                                                  H * W * 4 + H * W:,
    #                                                                                                  :]
    #     c = torch.cat([c_select1, c_select2 + x, c_select3], dim=1)
    #     c2 = c[:, 0:c2.size(1), :]
    #     c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
    #     c4 = c[:, c2.size(1) + c3.size(1):, :]
    #     y = c3
    #     # print('c2',c2.shape,'c3',c3.shape,'c4',c4.shape)
    #
    #     # Orientation_Blocks
    #     ori_9_d = self.Ori_blocks(x, y)  # [batch_size, num_ori_tokens, 3d, 3d]
    #
    #     # feed to mlp head
    #     x = rearrange(ori_9_d, 'batch oris d_1 d_2-> batch (oris d_1 d_2)')
    #
    #     x = self.mlp_head(x)
    #
    #     pred = compute_rotation_matrix_from_ortho6d(x)
    #
    #     return pred, ori_9_d
    def forward(self, x):
        """
        TokenHPE pipeline
        feature extractor (ViT) -> Orientation_Blocks -> outputs in all regions
        -> MLP head -> prediction: [pred, ori_9_d]
        """
        # feature extractor (ViT)
        c1, c2, c3, c4 = self.spm(x)
        x = self.feature_extractor(x)  # outputs: [batch_size, channel=197, dim = 768]

        c = torch.cat([c2, c3, c4], dim=1)
        c = self.mrfp(c, 14, 14)
        H = 14
        W = 14
        c_select1, c_select2, c_select3 = c[:, :H * W * 4, :], c[:, H * W * 4:H * W * 4 + H * W, :], c[:,
                                                                                                     H * W * 4 + H * W:,
                                                                                                     :]
        c = torch.cat([c_select1, c_select2 + x, c_select3], dim=1)
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]
        y = c3
        # print('c2',c2.shape,'c3',c3.shape,'c4',c4.shape)

        # Orientation_Blocks
        ori_9_d = self.Ori_blocks(x, y)  # [batch_size, 11, 3, 3]

        # feed to mlp head
        x = rearrange(ori_9_d, 'batch oris d_1 d_2-> batch (oris d_1 d_2)')

        x = self.mlp_head(x)

        pred = compute_rotation_matrix_from_ortho6d(x)

        return pred, ori_9_d
#
