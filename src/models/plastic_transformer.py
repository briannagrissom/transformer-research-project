import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_tensor(value: torch.Tensor) -> torch.Tensor:
    '''
    Ensures that the input is a tensor. If it is a scalar tensor, it is returned as is.
    If it is a multi-dimensional tensor, it is flattened to 1D
    '''
    if value.dim() == 0:
        return value
    return value.view(-1) # return as 1D tensor


@dataclass 
class PlasticLayerRecord:
    """Record of a single plastic layer's forward step for plasticity updates."""
    layer: "PlasticLinear" 
    state: Dict[str, torch.Tensor] # Plastic state for this layer
    pre: torch.Tensor # Pre-synaptic activity
    post: torch.Tensor # Post-synaptic activity
    require_grad: bool # Whether gradients were required for this step
    weight_grad: Optional[torch.Tensor] = None # Gradient for weight plasticity
    bias_grad: Optional[torch.Tensor] = None # Gradient for bias plasticity


class PlasticLinear(nn.Module):
    """
    Linear layer with a per-trial plastic component that is updated according to
    either a Hebbian or gradient-based plasticity rule.
    """

    def __init__(
        self,
        in_features: int, # Number of input features
        out_features: int, # Number of output features
        bias: bool = True, # Whether to include a bias term
        alpha_init: float = 0.02, # Initial value for alpha plasticity coefficient
        beta_init: float = 0.02, # Initial value for beta plasticity coefficient
    ) -> None:
        super().__init__() # Initialize the nn.Module
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features)) # Fixed weight matrix. torch.empty creates uninitialized tensor that will be initialized later
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features)) # Fixed bias vector
            self.register_parameter("bias", self.bias) # Register bias as a parameter
        else:
            self.bias = None
        self.alpha = nn.Parameter(torch.full((out_features, in_features), alpha_init)) # Plasticity coefficient for weights
        if bias:
            self.beta = nn.Parameter(torch.full((out_features,), beta_init)) # Plasticity coefficient for bias
        else:
            self.beta = None
        self.reset_parameters() # Initialize parameters

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # Kaiming uniform initialization for weights. This is suitable for layers with ReLU activations
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight) # Calculate fan-in for bias initialization. This is the number of input units to the layer
            bound = 1 / math.sqrt(fan_in) # Calculate bound for uniform distribution
            nn.init.uniform_(self.bias, -bound, bound) # Initialize bias uniformly within the calculated bounds. This helps to keep the initial outputs of the layer balanced

    def init_state(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Initializes the plastic state for this layer. State = {"w_plastic": ..., "b_plastic": ...}
        """
        state: Dict[str, torch.Tensor] = {
            "w_plastic": torch.zeros(
                self.out_features, self.in_features, device=device, dtype=self.weight.dtype # Plastic weight component initialized to zero. Shape: (out_features, in_features)
            ),
        }
        if self.bias is not None:
            state["b_plastic"] = torch.zeros(
                self.out_features, device=device, dtype=self.weight.dtype # Plastic bias component initialized to zero. Shape: (out_features,)
            )
        return state

    def forward_step(
        self,
        x: torch.Tensor,
        state: Dict[str, torch.Tensor],
        require_grad: bool = False,
    ) -> Tuple[torch.Tensor, PlasticLayerRecord]:
        """
        Processes a single time step.

        Args:
            x: Input vector of shape (in_features,)
            state: Plastic state dictionary for this layer.
            require_grad: Whether the plastic tensors should require gradients
                for gradient-based plasticity.

        Returns:
            Tuple containing the output vector and the record for later plastic updates.
        """
        if x.dim() != 1:
            raise ValueError("PlasticLinear.forward_step expects 1D input tensor.")

        if require_grad:
            state["w_plastic"] = state["w_plastic"].detach().requires_grad_(True) # Ensure plastic weights require gradients if needed
            if self.bias is not None:
                state["b_plastic"] = state["b_plastic"].detach().requires_grad_(True) # Ensure plastic bias requires gradients if needed

        total_weight = self.weight + state["w_plastic"] # Total effective weight is fixed + plastic component
        bias = None
        if self.bias is not None:
            bias = self.bias + state["b_plastic"] # Total effective bias is fixed + plastic component
        y = F.linear(x.unsqueeze(0), total_weight, bias=bias).squeeze(0) # Compute linear transformation. This outputs a vector of shape (out_features,).
        # y is the output vector after applying the linear transformation, which is the output of this layer at the current time step.
        # y = x @ total_weight.T + bias

        record = PlasticLayerRecord( # Create a record for this forward step
            layer=self, # reference to this layer
            state=state, # plastic state at this step
            pre=x, # activity before the layer
            post=y, # activity after the layer
            require_grad=require_grad,
        )
        return y, record # Return output and record

    def compute_hebbian_delta(self, record: PlasticLayerRecord) -> torch.Tensor:
        """
        Computes the Hebbian delta for this layer based on the provided record.
        Delta_w = outer(post, pre)
        """
        pre = record.pre.detach()
        post = record.post.detach()
        return torch.outer(post, pre) # return outer product of post and pre activities

    def apply_hebbian_update(
        self, record: PlasticLayerRecord, eta: torch.Tensor, delta: torch.Tensor
    ) -> None:
        """
        Applies the Hebbian plasticity update to the layer's plastic state.
        w_plastic = (1 - eta) * w_plastic + eta * alpha * delta
        b_plastic = (1 - eta) * b_plastic + eta * beta * post
        """
        with torch.no_grad():
            state = record.state
            state["w_plastic"].mul_(1.0 - eta).add_(eta * self.alpha * delta) # Update plastic weights
            # shape of eta is scalar, shape of alpha is (out_features, in_features), shape of delta is (out_features, in_features)
            if self.bias is not None:
                delta_b = record.post.detach() # Use post-synaptic activity for bias update
                beta = self.beta
                if beta is None:
                    raise RuntimeError("Bias plasticity requested but beta is None.")
                state["b_plastic"].mul_(1.0 - eta).add_(eta * beta * delta_b) # Update plastic bias
                # shape of eta is scalar, shape of beta is (out_features,), shape of delta_b is (out_features,)
        record.state["w_plastic"] = record.state["w_plastic"].detach() # Detach to prevent backprop through old graphs
        if self.bias is not None:
            record.state["b_plastic"] = record.state["b_plastic"].detach() # Detach to prevent backprop through old graphs

    def apply_gradient_update(
        self,
        record: PlasticLayerRecord,
        eta: torch.Tensor,
        weight_grad: Optional[torch.Tensor],
        bias_grad: Optional[torch.Tensor],
    ) -> None:
        with torch.no_grad():
            state = record.state
            if weight_grad is None:
                weight_grad = torch.zeros_like(state["w_plastic"])
            state["w_plastic"].mul_(1.0 - eta).add_(eta * self.alpha * weight_grad)
            if self.bias is not None:
                if bias_grad is None:
                    bias_grad = torch.zeros_like(state["b_plastic"])
                beta = self.beta
                if beta is None:
                    raise RuntimeError("Bias plasticity requested but beta is None.")
                state["b_plastic"].mul_(1.0 - eta).add_(eta * beta * bias_grad)

        if record.require_grad:
            # Detach the tensors so they do not keep references to old graphs.
            record.state["w_plastic"] = record.state["w_plastic"].detach()
            if self.bias is not None:
                record.state["b_plastic"] = record.state["b_plastic"].detach()


class PlasticFeedForward(nn.Module):
    """ Feed-forward network with plastic linear layers.
    """
    def __init__(
        self,
        dim_model: int, # Dimension of the model, which is the input and output dimension of the feed-forward network
        dim_ff: int, # Dimension of the feed-forward layer, which is the hidden layer size
        dropout: float, # Dropout rate
    ) -> None:
        super().__init__() # super means calling the constructor of the parent class nn.Module
        self.fc1 = PlasticLinear(dim_model, dim_ff) # First plastic linear layer
        self.fc2 = PlasticLinear(dim_ff, dim_model) # Second plastic linear layer
        self.act = nn.GELU() # GELU activation function, which is commonly used in transformer architectures
        self.dropout = nn.Dropout(dropout) # Dropout layer for regularization

    def init_state(self, device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "fc1": self.fc1.init_state(device), # Initialize plastic state for first layer
            "fc2": self.fc2.init_state(device), # Initialize plastic state for second layer
        }

    def forward_step(
        # This FF NN has two plastic linear layers with a GELU activation in between
        self,
        x: torch.Tensor, # Input tensor of shape (dim_model,), which is the input to the feed-forward network
        state: Dict[str, Dict[str, torch.Tensor]], # Plastic state dictionary for the feed-forward network
        require_grad: bool, # Whether gradients are required for plasticity updates
    ) -> Tuple[torch.Tensor, List[PlasticLayerRecord]]: # Output tensor and list of plastic layer records
        records: List[PlasticLayerRecord] = [] # List to store plastic layer records
        hidden1, rec1 = self.fc1.forward_step(x, state["fc1"], require_grad=require_grad) # Forward step through first plastic linear layer
        records.append(rec1) # Append record from first layer
        hidden1 = self.act(hidden1) # Apply GELU activation
        hidden1 = self.dropout(hidden1) # Apply dropout
        hidden2, rec2 = self.fc2.forward_step(hidden1, state["fc2"], require_grad=require_grad) # Forward step through second plastic linear layer
        records.append(rec2) # Append record from second layer
        hidden2 = self.dropout(hidden2) # Apply dropout
        return hidden2, records # return output and records


class PlasticTransformerBlock(nn.Module):
    """ Transformer block with plastic feed-forward network.
    """
    def __init__(
        self,
        dim_model: int, # Dimension of the model, which is the input and output dimension of the transformer block
        num_heads: int, # Number of attention heads
        dim_ff: int, # Dimension of the feed-forward layer
        dropout: float, # Dropout rate
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim_model) # Layer normalization before self-attention
        self.self_attn = nn.MultiheadAttention( # Multi-head self-attention layer
            embed_dim=dim_model, # Embedding dimension
            num_heads=num_heads, # Number of attention heads
            dropout=dropout, # Dropout rate
            batch_first=True, # Input and output tensors are provided as (batch, seq, feature)
        )
        self.dropout_attn = nn.Dropout(dropout) # Dropout layer after attention
        self.ln2 = nn.LayerNorm(dim_model) # Layer normalization before feed-forward network
        self.feed_forward = PlasticFeedForward(dim_model, dim_ff, dropout) # Feed-forward network with plastic layers

    def init_state(self, device: torch.device) -> Dict[str, Any]:
        return {
            "history": None,  # (1, steps, dim_model)
            "ff": self.feed_forward.init_state(device), # Plastic state for feed-forward network
        }

    def forward_step(
        self,
        x: torch.Tensor, # Input tensor of shape (dim_model,)
        state: Dict[str, Any], # Plastic state dictionary for the transformer block
        require_grad: bool, # Whether gradients are required for plasticity updates
    ) -> Tuple[torch.Tensor, List[PlasticLayerRecord]]: # Output tensor and list of plastic layer records
        residual = x # Save residual for skip connection
        x_norm = self.ln1(x) # Apply layer normalization
        q = x_norm.unsqueeze(0).unsqueeze(0)  # (batch=1, seq=1, dim)

        history = state["history"] # Retrieve history of past inputs for attention
        if history is None: # If no history, use current input as key/value
            kv = q
        else: # Concatenate history with current input for key/value
            kv = torch.cat([history, q], dim=1)

        attn_output, _ = self.self_attn(q, kv, kv, need_weights=False) # Self-attention computation. This computes attention of q over kv
        attn_output = attn_output.squeeze(0).squeeze(0) # Remove batch and seq dimensions
        x = residual + self.dropout_attn(attn_output) # Add skip connection and apply dropout
        state["history"] = kv # Update history with current input

        residual = x # Save residual for skip connection
        x_norm = self.ln2(x) # Apply layer normalization
        ff_out, records = self.feed_forward.forward_step( # Forward step through feed-forward network
            x_norm, state["ff"], require_grad=require_grad
        )
        x = residual + ff_out # Add skip connection
        return x, records # return output and records


class PlasticTransformerModel(nn.Module):
    """ Transformer model with plasticity in the feed-forward networks.
    """
    def __init__(
        self,
        input_dim: int, # Dimension of the input features
        output_dim: int, # Dimension of the output features
        model_dim: int = 128, # Dimension of the model (number of features in the transformer, which is also the input/output dimension of each transformer block)
        num_heads: int = 4, # Number of attention heads (multi-head attention)
        num_layers: int = 2, # Number of transformer layers (blocks)
        ffn_dim: int = 256, # Dimension of the feed-forward network (hidden layer size)
        dropout: float = 0.1, # Dropout rate
        aux_dim: int = 4, # Dimension of the auxiliary output
        rule: str = "hebbian", # Plasticity rule
        eta0: float = 0.2, # Initial learning rate
        max_norm: float = 1.0, # Maximum norm for gradient clipping
    ) -> None:
        super().__init__()
        if rule not in {"hebbian", "gradient", "none"}:
            raise ValueError(f"Unsupported plasticity rule: {rule}")
        self.rule = rule
        self.aux_dim = aux_dim
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, model_dim) # Input projection layer to model dimension. This maps input features to the model dimension
        self.blocks = nn.ModuleList(
            [
                PlasticTransformerBlock(model_dim, num_heads, ffn_dim, dropout) # Create transformer blocks
                for _ in range(num_layers) # Repeat for the number of layers
            ]
        )
        self.final_ln = nn.LayerNorm(model_dim) # Final layer normalization
        self.head = nn.Linear(model_dim, output_dim + aux_dim + 1) # Output head producing logits, auxiliary outputs, and tilde_eta
        self.eta0 = eta0 # Initial learning rate for plasticity
        self.max_norm = max_norm # Maximum norm for gradient clipping
        if rule == "gradient":
            internal_dim = output_dim + aux_dim + 1 # Dimension of the internal projection
            self.internal_proj = nn.Linear(internal_dim, internal_dim, bias=False) # Internal projection layer for gradient-based plasticity
        else:
            self.internal_proj = None # No internal projection for other rules

    def init_state(self, device: torch.device) -> Dict[str, Any]:
        return {
            "blocks": [block.init_state(device) for block in self.blocks], # Initialize plastic states for all transformer blocks
            "step": 0, # Initialize step counter
        }

    def forward_step( # Process a single time step, which means to process a single input sample
        self,
        x: torch.Tensor, # Input tensor of shape (input_dim,)
        state: Dict[str, Any], # Plastic state dictionary for the transformer model
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single time step (single sample). The caller is responsible for
        resetting plastic states between trials.
        """
        state["step"] += 1 # Increment step counter
        hidden = self.input_proj(x) # Project input to model dimension

        plastic_records: List[PlasticLayerRecord] = []
        for block, block_state in zip(self.blocks, state["blocks"]): # Iterate over transformer blocks and their states
            hidden, block_records = block.forward_step( # Process input through transformer block
                hidden, block_state, require_grad=self.rule == "gradient" # Require gradients only for gradient-based plasticity
            )
            plastic_records.extend(block_records) # Collect plastic layer records

        hidden = self.final_ln(hidden) # Apply final layer normalization
        raw_output = self.head(hidden) # Compute final output logits
        logits = raw_output[: self.output_dim] # Extract logits from output
        tilde_eta = raw_output[self.output_dim] # Extract tilde_eta from output, which is used to compute the plasticity learning rate
        aux = raw_output[self.output_dim + 1 :] # Extract auxiliary outputs from output, which are not used for plasticity updates

        if self.rule == "gradient":
            assert self.internal_proj is not None # Ensure internal projection is defined
            concat_vec = torch.cat([logits, aux, tilde_eta.unsqueeze(0)], dim=0) # Concatenate logits, auxiliary outputs, and tilde_eta
            internal_vec = self.internal_proj(concat_vec) # Project concatenated vector to internal representation
            internal_loss = (internal_vec.pow(2).mean()) # Internal loss for gradient computation
            grad_targets: List[torch.Tensor] = [] # List to collect plastic tensors for gradient computation
            for record in plastic_records: # Iterate over plastic layer records
                grad_targets.append(record.state["w_plastic"]) # Collect plastic weights
                if record.layer.bias is not None:
                    grad_targets.append(record.state["b_plastic"]) # Collect plastic bias if present
            grads = torch.autograd.grad( # Compute gradients of internal loss w.r.t. plastic tensors
                internal_loss,
                grad_targets,
                retain_graph=True,
                allow_unused=True,
            )
            grad_iter = iter(grads)
            delta_norm_sq = torch.tensor(0.0, device=logits.device)
            for record in plastic_records:
                weight_grad = next(grad_iter, None)
                bias_grad = None
                if record.layer.bias is not None:
                    bias_grad = next(grad_iter, None)
                record.weight_grad = (
                    torch.zeros_like(record.state["w_plastic"]) if weight_grad is None else weight_grad
                )
                record.bias_grad = (
                    torch.zeros_like(record.state["b_plastic"])
                    if record.layer.bias is not None and bias_grad is None
                    else bias_grad
                )
                delta_norm_sq = delta_norm_sq + record.weight_grad.pow(2).sum()
            delta_norm = torch.sqrt(delta_norm_sq + 1e-8)
        elif self.rule == "hebbian":
            delta_norm_sq = torch.tensor(0.0, device=logits.device) # Initialize squared norm of deltas
            for record in plastic_records: # Iterate over plastic layer records
                delta = record.layer.compute_hebbian_delta(record) # Compute Hebbian delta
                record.weight_grad = delta  # reuse field for update 
                delta_norm_sq = delta_norm_sq + delta.pow(2).sum() # Accumulate squared norm of deltas
                if record.layer.bias is not None: # If bias plasticity is present
                    record.bias_grad = record.post.detach() # reuse field for update
            delta_norm = torch.sqrt(delta_norm_sq + 1e-8) # Compute norm of deltas
        else:  # rule == "none"
            for record in plastic_records: # Iterate over plastic layer records
                record.state["w_plastic"] = record.state["w_plastic"].detach() # Detach plastic weights to prevent updates
                if record.layer.bias is not None:
                    record.state["b_plastic"] = record.state["b_plastic"].detach() # Detach plastic bias to prevent updates
            eta = torch.tensor(0.0, device=logits.device) # No plasticity learning rate
            return {
                "logits": logits, # Output logits
                "tilde_eta": tilde_eta.detach(), # Detach tilde_eta
                "aux": aux, # Auxiliary outputs
                "eta": eta, # Plasticity learning rate
                "diagnostics": {
                    "plastic_norm": torch.tensor(0.0, device=logits.device), # No plastic norm since no plasticity
                },
            }

        scaling = torch.ones_like(tilde_eta) # Initialize scaling factor for learning rate
        if self.max_norm > 0: # If max norm is set
            scaling = torch.clamp(self.max_norm / (delta_norm + 1e-8), max=1.0) # Compute scaling factor to limit norm of updates

        eta = self.eta0 * torch.sigmoid(tilde_eta) * scaling # Compute plasticity learning rate

        for record in plastic_records:
            if self.rule == "gradient":
                record.layer.apply_gradient_update(record, eta, record.weight_grad, record.bias_grad) # Apply gradient-based plasticity update
            else:
                record.layer.apply_hebbian_update(record, eta, record.weight_grad) # Apply Hebbian plasticity update

        plastic_norm = torch.tensor(0.0, device=logits.device) # Initialize plastic norm
        with torch.no_grad(): # Compute total plastic norm for diagnostics
            for block_state in state["blocks"]: # Iterate over transformer block states
                ff_state = block_state["ff"] # Get feed-forward plastic state
                for layer_state in ff_state.values(): # Iterate over plastic layers
                    w_plastic = layer_state.get("w_plastic") # Get plastic weights
                    if w_plastic is not None: # If plastic weights exist
                        plastic_norm = plastic_norm + w_plastic.pow(2).sum() # Accumulate squared norm

        output_dict = {
            "logits": logits, # Output logits
            "tilde_eta": tilde_eta.detach(), # Detach tilde_eta
            "aux": aux, # Auxiliary outputs
            "eta": eta.detach(), # Detach plasticity learning rate
            "diagnostics": {
                "plastic_norm": plastic_norm.detach(), # Detach plastic norm for diagnostics
            },
        }
        return output_dict
