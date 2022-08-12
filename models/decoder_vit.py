import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, kdim, vdim, 
                 dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, 
                                               dropout=dropout, 
                                               batch_first=batch_first,
                                               **factory_kwargs)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, 
                                                kdim=kdim, vdim=vdim, 
                                                dropout=dropout, 
                                                batch_first=batch_first,
                                                **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.activation = activation

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_pad_mask=None, memory_key_pad_mask=None, 
                short_cut=True):
        x = tgt
        if self.norm_first:
            x_, attn_weights = self._mha_block(self.norm1(x), memory, 
                                               memory_mask,
                                               memory_key_pad_mask)
            x = x + x_ if short_cut else x_
            x = x + self._sa_block(self.norm2(x), tgt_mask, tgt_key_pad_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x_, attn_weights = self._mha_block(x, memory, 
                                               memory_mask, 
                                               memory_key_pad_mask)
            x = x + x_ if short_cut else x_
            x = self.norm1(x)
            x = self.norm2(x + self._sa_block(x, tgt_mask, tgt_key_pad_mask))
            x = self.norm3(x + self._ff_block(x))

        return x, attn_weights

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout(x)

    # multihead attention block
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        output = self.cross_attn(x, mem, mem,
                                 attn_mask=attn_mask,
                                 key_padding_mask=key_padding_mask)
        x, attn_weights = output
        return self.dropout(x), attn_weights

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout(x)


class TransformerDecoder(nn.TransformerDecoder):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_pad_mask=None, memory_key_pad_mask=None):
        x = tgt

        attn_weights_all = []
        for mod in self.layers:
            x, attn_weights = mod(x, memory, tgt_mask=tgt_mask, 
                                  memory_mask=memory_mask,
                                  tgt_key_pad_mask=tgt_key_pad_mask,
                                  memory_key_pad_mask=memory_key_pad_mask)
            attn_weights_all.append(attn_weights)

        if self.norm is not None:
            x = self.norm(x)
        return x, attn_weights_all


class ImageEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embedding = nn.Conv2d(in_channels=3, 
                                         out_channels=cfg.enc_dim,
                                         kernel_size=cfg.patch_size, 
                                         stride=cfg.patch_size, 
                                         bias=False)

        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches, cfg.enc_dim)
        position_ids = torch.arange(num_patches).expand((1, -1))
        self.register_buffer("position_ids", position_ids)

    def forward(self, pixel_values):
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = patch_embeds + self.position_embedding(self.position_ids)
        return embeddings


class DecoderViTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cfg.enc_dim,
                                               cfg.enc_nhead,
                                               dropout=cfg.enc_dropout,
                                               batch_first=False)
        self.ffn = nn.Sequential(nn.Linear(cfg.enc_dim,
                                           cfg.enc_ffndim),
                                 nn.GELU(),
                                 nn.Dropout(cfg.enc_dropout),
                                 nn.Linear(cfg.enc_ffndim,
                                           cfg.enc_dim))
        self.cross_attn = nn.MultiheadAttention(cfg.enc_dim,
                                               cfg.enc_nhead,
                                               dropout=cfg.enc_dropout,
                                               kdim=cfg.dec_dim,
                                               vdim=cfg.dec_dim,
                                               batch_first=True)
        self.dropout = nn.Dropout(cfg.enc_dropout)
        self.norm1 = nn.LayerNorm(cfg.enc_dim, eps=cfg.layer_norm_eps)
        self.norm2 = nn.LayerNorm(cfg.enc_dim, eps=cfg.layer_norm_eps)
        self.norm3 = nn.LayerNorm(cfg.enc_dim, eps=cfg.layer_norm_eps)

        decoder_layer = TransformerDecoderLayer(cfg.dec_dim,
                                                cfg.dec_nhead,
                                                kdim=cfg.enc_dim,
                                                vdim=cfg.enc_dim,
                                                dim_feedforward=cfg.dec_ffn_dim,
                                                dropout=cfg.dec_dropout,
                                                activation=F.gelu,
                                                batch_first=True,
                                                norm_first=False)
        self.decoder_layers = TransformerDecoder(decoder_layer, 
                                                 cfg.num_layers_per_block)

    def forward(self, patch_emb, query):
        """
        Args:
            patch_emb: (B, L, D)
            query: (B, l, D), normalized
        Returns:
            patch_emb: (B, L, D)
            query: (B, l, D), normalized
            attn_weights
        """
        x = patch_emb
        x_ = self.norm1(x)
        x_ = self.self_attn(x_, x_, x_, need_weights=False)[0]
        x_ = self.dropout(x_)
        x = x + x_

        x_ = self.norm2(x)
        x_ = self.ffn(x_)
        x_ = self.dropout(x_)
        x = x + x_

        x_ = self.norm3(x)
        query, attn_weights = self.decoder_layers(tgt=query, memory=x_)
        x_ = self.cross_attn(x_, query, query, need_weights=False)[0]
        x_ = self.dropout(x_)
        patch_emb = x + x_

        return patch_emb, query, attn_weights


class DecoderViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embedding = ImageEmbeddings(cfg)
        self.query_embedding = nn.Embedding(cfg.num_queries, cfg.dec_dim)
        block = DecoderViTBlock(cfg)
        self.backbone = nn.ModuleList(_get_clones(block, cfg.num_blocks))
        self.layernorm = nn.LayerNorm(cfg.enc_dim, cfg.layer_norm_eps)
        self.classifier = nn.Linear(cfg.enc_dim, cfg.num_classes)

        self.cfg = cfg
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, img):
        bs = img.size(0)
        patch_emb = self.patch_embedding(img)
        query = self.query_embedding.weight[None]
        query = query.repeat(bs, 1, 1)

        attns = []
        for block in self.backbone:
            patch_emb, query, attn = block(patch_emb, query)
            attns.append(attn)

        patch_emb = self.layernorm(patch_emb)
        avg_patch_emb = patch_emb.mean(dim=1)
        logits = self.classifier(avg_patch_emb)

        return logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
