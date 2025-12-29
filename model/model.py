from transformers import PretrainedConfig


class MindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        
        ############ MoE ############
        use_moe:bool=False,
        num_experts_per_tok:int=2,
        n_routed_experts:int=4,
        n_shared_experts:int=1,
        scoring_func:str='softmax',
        aux_loss_alpha:float=0.1,
        seq_aux:bool=True,
        norm_topk_prob:bool=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe=use_moe
        self.num_experts_per_tok=num_experts_per_tok
        self.n_routed_experts=n_routed_experts
        self.n_shared_experts=n_shared_experts
        self.seq_aux=seq_aux
        self.norm_topk_prob=norm_topk_prob
        self.aux_loss_alpha=aux_loss_alpha
        self.scoring_func=scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import math
import torch.nn as nn
from typing import Optional, Tuple,List,Union
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

# Inherit the nn.Module class
class RMSNorm(nn.Module):
# __init__
    def __init__(self, dim:int, eps:float=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
# _norm
    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
# forward
    def forward(self, x):
        return x * self.weight * self._norm(x.float()).type_as(x)
    
def precompute_freqs_cis(dim:int, end:int=int(32*1024), rope_base:flot=1e6,
                         rope_scaling:Optional[dict]=None):
# Write the initial RoPE formula
    freqs = 1.0 / rope_base ** torch.arange(0, dim, 2)[:dim//2].float()/dim
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))

    if rope_scaling is not None:
            original_max, factor, beta_fast, beta_slow = (
                rope_scaling.get("original_max_position_embeddings", 2048),
                rope_scaling.get("factor", 4),
                rope_scaling.get("beta_fast", 4.0),
                rope_scaling.get("beta_slow", 1.0),
            )

            if end / original_max > 1.0:
                # Calculate corr_dim
                corr_dim = next(
                    (i for i in range(dim // 2) if 2 * math.pi / freqs[i] > original_max),
                    dim // 2,
                )
                # Calculate power
                power = torch.arange(0, dim // 2, device=freqs.device).float() / max(
                    dim // 2 - 1, 1
                )

                # Calculate beta
                beta = beta_slow + (beta_fast - beta_slow) * power

                # CalculateScale
                scale = torch.where(
                    torch.arange(dim // 2, device=freqs.device) < corr_dim,
                    (beta * factor - beta + 1) / (beta * factor), # high frequency
                    1.0 / factor, # low frequency
                )

                # ApplyScale
                freqs = freqs * scale

            # Generate position index, multiplied by frequency
            t = torch.arange(end, device=freqs.device)
            freqs = torch.outer(t, freqs).float() # [end, dim//2]

            # Returns a cos and sin
            freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
            freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)

            return freqs_cos, freqs_sin
    
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    # [a,b] -> [b,a]
    def rotate_half(x):
        # x.shape[-1]Take the end point of the last dimension
        # x[..., x.shape[-1] // 2 :]Take the second half
        # x[..., ：， x.shape[-1] // 2]Take the first half
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    # Apply rotation encoding
    # x_rotated = x * cos + rotate_half(x) * sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args:MindConfig):
        super().__init__()

        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads

        # Evenly group the H query heads into G K/V heads.
        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        ## repeat count, how many Q heads are shared by each K/V head
        self.n_rep = self.n_local_heads // self.n_local_kv_heads 

        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        ## Check whether FlashAttention is enabled; if True, use F.scaled_dot_product_attention in the forward method (which internally dispatches to FlashAttention-2 or cuDNN SDPA when available), significantly accelerating computation and reducing memory usage. Otherwise, fall back to the manual implementation of softmax(QK^T)V.
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # Projection, calculate q, k, v
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # Split the input into multiple heads (use view)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        # q and k, use RoPE
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # For k and v, use repeat
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        past_kv = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)
        # [bsz, n_local_heads, seq_len, head_dim]
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # Perform attention calculation
        if (
            self.flash
            and seq_len > 1
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1, -1)
                .expand(bsz, self.n_local_heads, seq_len, -1)
                .bool()
            )
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,  # Autoregressive (causal) attention
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=-1,
            )

            scores=scores+causal_mask.unsqueeze(0).unsqueeze(0)

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores@xv

        # Finally, splice the header, output the projection, and return
        output = output.transpose(1,2).reshape(
            bsz,seq_len,-1
        )
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv