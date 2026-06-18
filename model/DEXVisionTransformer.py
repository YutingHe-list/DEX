from typing import Callable, Optional, Type
import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.layers import (
    Attention,
    Mlp,
    LayerNorm,
    DropPath,
)

import torch

def kd_loss_fn(
    f_experts, 
    f_director, 
    use_mse=False, 
    use_kd=False, 
    use_cos=False,
    T_student=1.0, 
    T_teacher=0.5, 
    eps=1e-6
):
    """
    f_experts: student features (B, D)
    f_director: teacher features (B, D)
    """
    
    if use_mse:
        return F.mse_loss(f_experts, f_director)

    elif use_kd:
        # stop gradient on teacher
        f_director = f_director.detach()

        # temperature scaling + normalized sigmoid
        q = torch.softmax(f_experts / T_student, dim=-1)

        p = torch.softmax(f_director / T_teacher, dim=-1)

        return -(p * torch.log(torch.clamp(q, min=eps))).sum(dim=-1).mean()

    elif use_cos:
        return 1.0 - F.cosine_similarity(f_experts, f_director, dim=-1).mean()

    else:
        raise ValueError("Specify one of use_mse / use_kd / use_cos.")

class LayerScale(nn.Module):
    """Layer scale module.

    References:
      - https://arxiv.org/abs/2103.17239
    """

    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        """Initialize LayerScale module.

        Args:
            dim: Dimension.
            init_values: Initial value for scaling.
            inplace: If True, perform inplace operations.
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer scaling."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

    
class NoisyTopKRouter(nn.Module):
    def __init__(self, 
                 in_features, 
                 num_experts, 
                 top_k=2, 
                 batchwise=True, 
                 tau=0.5,):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.linear = nn.Linear(in_features, num_experts, bias=False)
        self.batchwise = batchwise
        self.tau = tau

    def forward(self, x, expert_activation_count, noise_std=0.):
        """
        Args:
            x: [B, N, D] input features
            expert_activation_count: [num_experts] tensor tracking activation counts for load balancing
            noise_std: float, standard deviation of the noise for exploration (only during training)
        Returns:
            gates: [B, E, K] gating weights for selected experts
            indices: [B, E, K] indices of selected experts
            aux_loss: float, auxiliary load balancing loss
        """
        if self.batchwise:
            logits = self.linear(x[:, 0:1, :]) 
        else:
            logits = self.linear(x)     

        K = self.top_k

        # add noise for exploration during training
        if self.training and noise_std > 0:
            freq = 1.0 - expert_activation_count / (expert_activation_count.sum() + 1e-6)
            logits = logits + noise_std * (torch.randn_like(logits) + freq.view(1, 1, -1).expand_as(logits))

        # soft gating -> top-k candidates
        probs = torch.softmax(logits/self.tau, dim=-1) 
        scores, indices = torch.topk(probs, K, dim=-1)  
        gates = scores / scores.sum(dim=-1, keepdim=True)    
        
        # Load Balancing Loss
        importance = probs.mean(dim=(0, 1))        
        load       = importance.detach()           
        aux_loss   = self.num_experts * torch.sum(importance * load)

        return gates, indices, aux_loss

class DEX_layer(nn.Module):
    def __init__(self,
                 dim,
                 num_experts: int                       = 16,
                 mlp_hidden_dim: int                    = 4096,
                 top_k: int                             = 2,
                 drop: float                            = 0.0,
                 ffn_bias: bool                         = True,
                 scale_mlp_norm: bool                   = False,
                 act_layer: Callable[..., nn.Module]    = nn.GELU,
                 ffn_layer: Callable[..., nn.Module]    = Mlp,
                 norm_layer: Type[nn.Module]            = LayerNorm,
                 batchwise: bool                        = True,
                 switch_pre_ft: bool                    = True, # True for pre, False for FT
                 switch_experts_directors: bool         = True, # True for experts, False for director
                 ):
        super().__init__()
        """
        Args:
            dim: int, embedding dimension
            num_experts: int, number of experts in the DEX module
            mlp_hidden_dim: int, hidden dimension of the MLP in DEX
            top_k: int, number of experts to select for each token
            drop: float, dropout rate for the output of each expert
            ffn_bias: bool, whether to add bias in the MLP of DEX
            scale_mlp_norm: bool, whether to apply LayerNorm before MLP in DEX
            act_layer: Callable[..., nn.Module], activation layer for MLP in DEX
            ffn_layer: Callable[..., nn.Module], feedforward network layer for DEX (e.g., MLP)
            norm_layer: Type[nn.Module], normalization layer for DEX (e.g., LayerNorm)
            batchwise: bool, whether to perform routing at the image level (True) or token level (False)
            switch_pre_ft: bool, whether to use pre-training mode (True) or fine-tuning mode (False)
            switch_experts_directors: bool, only relevant if switch_pre_ft=False, train experts (True) or director (False) during fine-tuning
        Returns:
            output: Tensor of shape [B, N, D], the result of the DEX module
            co_loss: float, the co-training loss between experts and director (only in pre-training mode)
            bal_loss: float, the load balancing loss for expert routing (only in pre-training mode)
        """
        # prepare parameter
        self.mlp_hidden_dim = mlp_hidden_dim
        self.batchwise = batchwise
        self.switch_pre_ft = switch_pre_ft
        self.top_k = top_k

        # Director
        self.director = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=ffn_bias,
            drop=drop,
        )

        # Experts
        self.experts = nn.ModuleList([ffn_layer(in_features=dim,
                                                hidden_features=mlp_hidden_dim,
                                                act_layer=act_layer,
                                                norm_layer=norm_layer if scale_mlp_norm else None,
                                                bias=ffn_bias,
                                                drop=drop,
                                                ) for _ in range(num_experts)])      
        # router for experts
        self.router = NoisyTopKRouter(in_features=dim, num_experts=num_experts, top_k=top_k, batchwise=batchwise)
        
        # prepare state
        if self.switch_pre_ft:
            for expert in self.experts:
                expert.load_state_dict(self.director.state_dict(), strict=True)
            for param in self.director.parameters():
                param.requires_grad = False  # not update by gradient
        else:
            self.switch_experts_directors = switch_experts_directors
            for param in self.parameters():
                param.requires_grad = True  # release all for gradient

        self.register_buffer(
            "expert_activation_count",
            torch.zeros(num_experts, dtype=torch.float32)
        )

        self.use_momentum_stats = True  # or False

        self.momentum = 0.99

    @torch.no_grad()
    def _momentum_update_director(self, gates, indices, m: float = 0.999) -> None:
        """
        Momentum update of the director using top-k selected experts.
        
        Args:
            gates (Tensor): [B, 1(N), K] gating weights
            indices (Tensor): [B, 1(N), K] selected expert indices
            m (float): momentum factor (e.g., 0.999)
        """
        num_experts = len(self.experts)
        device = gates.device

        # calculate expert weights by summing gates of assigned tokens
        expert_weights = torch.zeros(num_experts, device=device)
        flat_indices = indices.view(-1)    
        flat_gates = gates.view(-1)        
        expert_weights.scatter_add_(0, flat_indices, flat_gates)

        # normalize to get a distribution over experts
        expert_weights = expert_weights / (expert_weights.sum() + 1e-6)  # [E]

        # momentum update director parameters towards the weighted average of selected experts
        with torch.no_grad():
            for name, director_param in self.director.named_parameters():
                # collect corresponding parameters from all experts
                expert_params = torch.stack(
                    [dict(expert.named_parameters())[name].data for expert in self.experts],
                    dim=0  # [num_experts, *param_shape]
                )

                # weighted average of expert parameters
                w = expert_weights.view(-1, *([1] * (expert_params.dim() - 1)))  # broadcast
                weighted_avg = torch.sum(w * expert_params, dim=0)

                # update director parameter with momentum
                director_param.data = m * director_param.data + (1 - m) * weighted_avg

    @torch.no_grad()
    def forward_director(self, x):
        # x: [B, N, D] (batch, tokens, dim)
        out = self.director(x)  # [B, N, D]
        return out

    def forward_experts(self, x, noise_std=0.):
        # x: [B, N, D] (batch, tokens, dim)
        B, N, D = x.shape
        gates, indices, bal_loss = self.router(x, self.expert_activation_count, noise_std)  #  gates: [B, E, K], indices: [B, E, K]
        B, E, K = gates.shape # E == N or E == 1
        device = x.device
        
        # Flatten for convenience
        if self.batchwise:
            # image-wise: treat each image as one unit (T = B)
            T = B
            indices_flat = indices.view(T, K)    # [B, K]
            gates_flat = gates.view(T, K)        # [B, K]
            out_images = torch.zeros(B, N, D, device=device)
        else:
            # token-wise
            T = B * E
            x_flat = x.reshape(B * N, D)
            indices_flat = indices.reshape(T, K)
            gates_flat = gates.reshape(T, K)
            out_flat = torch.zeros(B * N, D, device=device)
        
        # Update expert activation statistics
        with torch.no_grad():
            num_experts = len(self.experts)
            device = self.expert_activation_count.device

            # add up the gates for each expert to get the current activation scores
            current_scores = torch.zeros(num_experts, device=device)

            # indices_flat: [T, K]
            # gates_flat:   [T, K]
            flat_indices = indices_flat.view(-1)    # [T*K]
            flat_gates   = gates_flat.view(-1)      # [T*K]

            # scatter_add the gates to the corresponding expert indices
            current_scores.scatter_add_(0, flat_indices, flat_gates)

            # normalize to get a distribution over experts
            current_scores = current_scores / (current_scores.sum() + 1e-6)

            # momentum update
            if self.use_momentum_stats:
                self.expert_activation_count.mul_(self.momentum).add_((1 - self.momentum) * current_scores)
            else:
                self.expert_activation_count += current_scores

        
        #  Dispatch: group by expert
        for a, expert in enumerate(self.experts):
            selected = (indices_flat == a) & (gates_flat > 0)  # [T, K]
            if not selected.any():
                continue

            pred_idx, k_idx = torch.nonzero(selected, as_tuple=True)  # pred_idx: [num_assigned], k_idx: [num_assigned]
            gate = gates_flat[pred_idx, k_idx]  # [num_assigned]
            if self.batchwise:
                # expert_input: whole-image tokens for selected images
                expert_input = x[pred_idx]                    # [num_assigned, N, D]
                expert_output = expert(expert_input)            # expert must accept (..., N, D) -> [num_assigned, N, D]
                g = gate.view(-1, 1, 1)                      # broadcast over tokens and dim
                out_images[pred_idx] += expert_output * g
            else:
                x_flat = x.reshape(B * N, D)
                expert_input = x_flat[pred_idx]               # [num_assigned, D]
                expert_output = expert(expert_input)            # [num_assigned, D]
                g = gate.view(-1, 1)
                out_flat[pred_idx] += expert_output * g

        out = out_images if self.batchwise else out_flat.view(B, N, D)

        return out, gates, indices, bal_loss
    
    def forward(self, x, m: float = 0.999, noise_std: float=0.):
        if self.switch_pre_ft:
            f_experts, gates, indices, bal_loss = self.forward_experts(x, noise_std)
            f_directors = self.forward_director(x)
            co_loss = kd_loss_fn(f_experts, f_directors, use_kd=True)

            self._momentum_update_director(gates, indices, m)
            return f_experts, co_loss, bal_loss
        else:
            if self.switch_experts_directors:
                f, _, _, _ = self.forward_experts(x)
            else:
                f = self.forward_director(x)
            return f

class DEXBlock(nn.Module):
    """Director-Experts block."""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            num_experts: int = 16,
            top_k: int = 2,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_attn_norm: bool = False,
            scale_mlp_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            ffn_layer: Type[nn.Module] = Mlp,
            switch_pre_ft=True, # True for pre, False for FT
            switch_experts_directors=True, # True for experts, False for director
    ) -> None:
        """
        Args:
            dim: int, embedding dimension
            num_heads: int, number of attention heads
            num_experts: int, number of experts in the DEX module
            top_k: int, number of experts to select for each token
            mlp_ratio: float, expansion ratio for MLP hidden dimension
            qkv_bias: bool, whether to add bias in QKV projection
            qk_norm: bool, whether to apply normalization to Q and K in attention
            scale_attn_norm: bool, whether to apply LayerNorm before attention
            scale_mlp_norm: bool, whether to apply LayerNorm before MLP in DEX
            proj_bias: bool, whether to add bias in output projection of attention and MLP
            proj_drop: float, dropout rate for output projection
            attn_drop: float, dropout rate for attention weights
            init_values: Optional[float], if not None, the initial value for LayerScale
            drop_path: float, drop path rate for stochastic depth
            act_layer: Type[nn.Module], activation layer for MLP
            norm_layer: Type[nn.Module], normalization layer
            ffn_layer: Type[nn.Module], feedforward network layer (e.g., MLP)
            switch_pre_ft: bool, whether to use pre-training mode (True) or fine-tuning mode (False)
            switch_experts_directors: bool, only relevant if switch_pre_ft=False, train experts (True) or director (False) during fine-tuning
        Returns:
            output: Tensor of shape [B, N, D], the result of the DEX block
            co_loss: float, the co-training loss between experts and director (only in pre-training mode)
            bal_loss: float, the load balancing loss for expert routing (only in pre-training mode)
        """
        super().__init__()

        self.switch_pre_ft = switch_pre_ft
        # Attention part
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # DEX module part
        self.norm2 = norm_layer(dim)
        self.DEX = DEX_layer(
            dim=dim,
            num_experts=num_experts,
            mlp_hidden_dim=int(dim * mlp_ratio),
            top_k=top_k,
            drop=proj_drop,
            ffn_bias=proj_bias,
            scale_mlp_norm=scale_mlp_norm,
            act_layer=act_layer,
            ffn_layer=ffn_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            switch_pre_ft=switch_pre_ft,
            switch_experts_directors=switch_experts_directors
        )

        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, m: float=0.999, noise_std: float=0., alpha: float=1.):
        # Attention
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))

        # DEX
        if self.switch_pre_ft:
            f, co_loss, bal_loss = self.DEX(self.norm2(x), m, noise_std)
            x = x + self.drop_path2(self.ls2(f))
            return x, co_loss/alpha, bal_loss
        else:
            f = self.DEX(self.norm2(x))
            x = x + self.drop_path2(self.ls2(f))
            return x