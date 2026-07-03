"""CodeMaster 270M Parameter Model Architecture."""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import PreTrainedModel, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithPast


class CodeMasterConfig(GPT2Config):
    """Configuration for CodeMaster 270M model."""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 2048,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-12,
        initializer_range: float = 0.02,
        summary_type: str = "cls_index",
        summary_use_proj: bool = True,
        summary_proj_to_labels: bool = True,
        summary_first_dropout: float = 0.1,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_range=initializer_range,
            summary_type=summary_type,
            summary_use_proj=summary_use_proj,
            summary_proj_to_labels=summary_proj_to_labels,
            summary_first_dropout=summary_first_dropout,
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


class CodeMasterAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(self, config: CodeMasterConfig):
        super().__init__()
        self.num_heads = config.n_head
        self.attention_head_size = config.n_embd // config.n_head
        self.all_head_size = self.num_heads * self.attention_head_size
        
        self.query = nn.Linear(config.n_embd, self.all_head_size)
        self.key = nn.Linear(config.n_embd, self.all_head_size)
        self.value = nn.Linear(config.n_embd, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attn_pdrop)
        self.output_projection = nn.Linear(config.n_embd, config.n_embd)
        self.output_dropout = nn.Dropout(config.resid_pdrop)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for attention computation."""
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass through attention."""
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.tensor(
            self.attention_head_size**0.5, dtype=attention_scores.dtype
        )
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        batch_size = context_layer.size(0)
        context_layer = context_layer.view(batch_size, -1, self.all_head_size)
        
        output = self.output_projection(context_layer)
        output = self.output_dropout(output)
        
        outputs = (output,)
        if output_attentions:
            outputs = outputs + (attention_probs,)
        
        return outputs


class CodeMasterMLP(nn.Module):
    """Feed-forward network in transformer block."""
    
    def __init__(self, config: CodeMasterConfig):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.n_embd, config.n_inner or 4 * config.n_embd)
        self.dense_4h_to_h = nn.Linear(config.n_inner or 4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.activation = nn.GELU()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class CodeMasterBlock(nn.Module):
    """Single transformer block with attention and feed-forward."""
    
    def __init__(self, config: CodeMasterConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CodeMasterAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = CodeMasterMLP(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass through transformer block."""
        attn_output = self.attn(
            self.ln_1(hidden_states),
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        attn_output = attn_output[0]
        hidden_states = hidden_states + attn_output
        
        mlp_output = self.mlp(self.ln_2(hidden_states))
        hidden_states = hidden_states + mlp_output
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (attn_output[1],)
        
        return outputs


class CodeMasterModel(PreTrainedModel):
    """CodeMaster 270M Parameter Language Model for Programming.
    
    Architecture:
    - 12 transformer layers
    - 768 hidden dimensions
    - 12 attention heads
    - ~270M parameters
    - 2048 token context window
    """
    
    config_class = CodeMasterConfig
    
    def __init__(self, config: CodeMasterConfig):
        super().__init__(config)
        self.config = config
        
        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer blocks
        self.h = nn.ModuleList(
            [CodeMasterBlock(config) for _ in range(config.n_layer)]
        )
        
        # Layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Output projection
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.init_weights()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        """Forward pass of CodeMaster model."""
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        input_shape = input_ids.size()
        batch_size = input_shape[0]
        device = input_ids.device
        
        if position_ids is None:
            position_ids = torch.arange(
                0, input_shape[-1], dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        for block in self.h:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            
            hidden_states = outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)
        
        hidden_states = self.ln_f(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        
        if not return_dict:
            return tuple(
                v
                for v in [loss, lm_logits, all_hidden_states, all_self_attentions]
                if v is not None
            )
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
