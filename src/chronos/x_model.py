import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union, List
import logging
import json
import os
from transformers import AutoModel
from transformers.modeling_outputs import Seq2SeqLMOutput

from .x_wrapper import ModelWrapper, ChronosWrapper, ChronosBoltWrapper, TimesFMWrapper

logger = logging.getLogger(__name__)


class FFN(nn.Module):
    """Feed-Forward Network with ReLU activation"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class InputInjectionBlock(nn.Module):
    """
    Input Injection Block - adds past covariates to token embeddings
    gIIB(hemb(zt−1), xt−1) = FFN(ReLU(hemb(zt-1)W_IIB^(emb)) ⊕ xt−1W_IIB^(cov))
    """
    
    def __init__(self, d_model: int, covariate_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.d_model = d_model
        self.covariate_dim = covariate_dim
        
        self.W_emb = nn.Linear(d_model, hidden_dim)
        self.W_cov = nn.Linear(covariate_dim, hidden_dim)
        self.ffn = FFN(2 * hidden_dim, hidden_dim, d_model)
        
    def forward(self, token_embeddings: torch.Tensor, past_covariates: torch.Tensor = None) -> torch.Tensor:
        if past_covariates is None:
            return token_embeddings

        # Handle device placement
        if token_embeddings.device != past_covariates.device:
            past_covariates = past_covariates.to(token_embeddings.device)
        
        # Handle sequence length mismatch
        if token_embeddings.shape[1] != past_covariates.shape[1]:
            min_len = min(token_embeddings.shape[1], past_covariates.shape[1])
            token_embeddings = token_embeddings[:, :min_len]
            past_covariates = past_covariates[:, :min_len]
        
        # Project embeddings and covariates (separate linear transformations)
        emb_proj = self.W_emb(token_embeddings)  # hemb(zt−1)W_IIB^(emb)
        cov_proj = self.W_cov(past_covariates)   # xt−1W_IIB^(cov)
        
        # Apply ReLU to each projection separately, then concatenate
        emb_relu = F.relu(emb_proj)
        cov_relu = F.relu(cov_proj)
        concatenated = torch.cat([emb_relu, cov_relu], dim=-1)  # ⊕ operation
        
        # Apply FFN to get adjustment
        adjustment = self.ffn(concatenated)
        
        # Add to original embeddings (residual connection)
        # fIIB(zt−1, xt−1) = hemb(zt−1) + gIIB(hemb(zt−1), xt−1)
        return token_embeddings + adjustment


class OutputInjectionBlock(nn.Module):
    """
    Output Injection Block - adjusts outputs using future covariates
    fOIB(zt−1, xt) = hout(zt−1)Wout + gOIB(hout(zt−1), xt)
    """

    def __init__(self, d_model: int, covariate_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.d_model = d_model
        self.covariate_dim = covariate_dim
        self.output_dim = output_dim
        
        # For covariate injection
        self.W_out = nn.Linear(d_model, hidden_dim)
        self.W_cov = nn.Linear(covariate_dim, hidden_dim)
        self.ffn = FFN(2 * hidden_dim, hidden_dim, output_dim)
        
        # For forecasting head (when no future covariates)
        self.forecasting_head = nn.Linear(d_model, output_dim)

    def forward(self, hidden_states: torch.Tensor, future_covariates: torch.Tensor = None, 
                original_outputs: torch.Tensor = None) -> torch.Tensor:
        # Case 1: No future covariates - act as a forecasting head
        if future_covariates is None:
            return original_outputs

        # Case 2: Future covariates available - do injection
        # Handle device placement
        if hidden_states.device != future_covariates.device:
            future_covariates = future_covariates.to(hidden_states.device)
        
        # Handle different output shapes for different models
        if len(original_outputs.shape) == 3 and original_outputs.shape[1] != hidden_states.shape[1]:
            # For models that output single timestep, use last hidden state
            hidden_states = hidden_states[:, -1:, :]
            if future_covariates.shape[1] != 1:
                future_covariates = future_covariates.mean(dim=1, keepdim=True)
        
        # Project hidden states and covariates
        hidden_proj = self.W_out(hidden_states)     # hout(zt−1)W_OIB^(out)
        cov_proj = self.W_cov(future_covariates)    # xtW_OIB^(cov)
        
        # Apply ReLU and concatenate
        hidden_relu = F.relu(hidden_proj)
        cov_relu = F.relu(cov_proj)
        concatenated = torch.cat([hidden_relu, cov_relu], dim=-1)
        
        # Apply FFN to get adjustment
        adjustment = self.ffn(concatenated)  # gOIB(hout(zt−1), xt)
        
        # Reshape adjustment to match original_outputs if needed
        if adjustment.shape != original_outputs.shape:
            if len(original_outputs.shape) == 3 and original_outputs.shape[1] != adjustment.shape[1]:
                adjustment = adjustment.squeeze(1)
                batch_size = original_outputs.shape[0]
                adjustment = adjustment.view(batch_size, -1, original_outputs.shape[-1])
        
        # Add adjustment to original outputs to get the logits
        # fOIB(zt−1, xt) = hout(zt−1)Wout + gOIB(hout(zt−1), xt)
        return original_outputs + adjustment


class XAdaptor(nn.Module):
    """Universal covariate adapter that can be attached to any pretrained model"""
    
    def __init__(self,
                 d_model: int,
                 covariate_dim: int,
                 output_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.covariate_dim = covariate_dim
        
        # Initialize injection blocks based on configuration
        self.input_injection = InputInjectionBlock(d_model, covariate_dim, hidden_dim)
        self.output_injection = OutputInjectionBlock(d_model, covariate_dim, output_dim, hidden_dim)
    
    def inject_input_covariates(self, token_embeddings: torch.Tensor, 
                               past_covariates: torch.Tensor) -> torch.Tensor:
        """Apply Input Injection Block"""
        return self.input_injection(token_embeddings, past_covariates)
    
    def inject_output_covariates(self, hidden_states: torch.Tensor,
                                future_covariates: torch.Tensor,
                                original_outputs: torch.Tensor) -> torch.Tensor:
        """Apply Output Injection Block"""
        return self.output_injection(hidden_states, future_covariates, original_outputs)


# Universal model class
class AdaptedXModel(nn.Module):
    """Universal model that can add covariates to any time series model"""

    def __init__(self,
                 model_wrapper: ModelWrapper,
                 covariate_dim: int,
                 hidden_dim: int = 256,
                 freeze_pretrained: bool = True):
        super().__init__()

        self.model_wrapper = model_wrapper
        self.freeze_pretrained = freeze_pretrained
        self.model_type = model_wrapper.model_type
        self.covariate_dim = covariate_dim
        self.hidden_dim = hidden_dim

        # Freeze pretrained model parameters if specified
        if freeze_pretrained:
            if hasattr(self.model_wrapper._get_base_model(), 'parameters'):
                for param in self.model_wrapper._get_base_model().parameters():
                    param.requires_grad = False
            elif hasattr(self.model_wrapper._get_base_model(), 'model'): # edge case for ChronosPipeline
                for param in self.model_wrapper._get_base_model().model.parameters():
                    param.requires_grad = False
            else:
                raise ValueError("Unsupported model type. Model does not have parameters?")

        # Initialize covariate adapter
        self.covariate_adapter = XAdaptor(
            d_model=model_wrapper.d_model,
            covariate_dim=covariate_dim,
            output_dim=model_wrapper.output_dim,
            hidden_dim=hidden_dim,
        )

        # Store base model path for reconstruction
        self._base_model_path = None

    def forward(self, 
                input_data: torch.Tensor,
                labels: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                past_covariates: Optional[torch.Tensor] = None,
                future_covariates: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Universal forward pass"""
        # Ensure all inputs are on the same device
        device = input_data.device
        if mask is not None:
            mask = mask.to(device)
        if past_covariates is not None:
            past_covariates = past_covariates.to(device)
        if future_covariates is not None:
            future_covariates = future_covariates.to(device)

        # Get initial embeddings
        embeddings = self.model_wrapper.get_input_embeddings(input_data, **kwargs)

        # Apply Input Injection Block
        embeddings = self.covariate_adapter.inject_input_covariates(embeddings, past_covariates)

        # Forward pass through pretrained model
        hidden_states, logits = self.model_wrapper.forward_with_embeddings(embeddings, labels, mask, **kwargs)

        # Apply Output Injection Block if future covariates provided
        # *** We currently don't want to use future covariates
        if future_covariates is not None:
            logits = self.covariate_adapter.inject_output_covariates(hidden_states, future_covariates, logits)

        # Calculate loss
        loss = self.model_wrapper.compute_loss(logits)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
        )

    def generate(self, input_data: torch.Tensor, 
                 labels: Optional[torch.Tensor] = None,
                 past_covariates: Optional[torch.Tensor] = None,
                 future_covariates: Optional[torch.Tensor] = None,
                 num_samples: int = 20,
                 max_length: int = 64,
                 **generation_kwargs) -> torch.Tensor:
        """Generate outputs using the model"""
        if self.model_type == 'chronosbolt':
            # TODO: implement generate_chronosbolt()
            return self.generate_chronosbolt(input_data, labels, past_covariates, future_covariates, num_samples, max_length, **generation_kwargs)
        elif self.model_type != 'chronos':
            return self.forward(input_data, labels, past_covariates, future_covariates, **generation_kwargs)

        input_data = input_data.to(self.model_wrapper.tokenizer.boundaries.device)
        input_tensor = self.model_wrapper.chronos_pipeline._prepare_and_validate_context(context=input_data)
        transformed = self.model_wrapper.tokenizer.context_input_transform(input_tensor)
        input_ids, attention_mask, scale = [t.to(self.model_wrapper.model.device) for t in transformed]

        batch_size = input_ids.size(0)
        device = input_ids.device

        # If no adapters or covariates, use standard generation
        if past_covariates is None and future_covariates is None:
            print("Using standard generation without covariates")
            generated_ids = self.model_wrapper.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prediction_length=max_length,
                num_samples=num_samples,
                **generation_kwargs,
            )

            # Convert generated IDs back to tokens
            predictions = self.model_wrapper.tokenizer.output_transform(
                generated_ids.to("cpu"), scale.to("cpu")
            )

            return predictions

        # Step 1: Process encoder with input injection if available - assumed always available
        encoder_embeddings = self.model_wrapper.model.model.encoder.embed_tokens(input_ids)
        modified_embeddings = self.covariate_adapter.inject_input_covariates(
            encoder_embeddings, past_covariates
        )
        modified_embeddings = self.model_wrapper.model.model.encoder.dropout(modified_embeddings)
        encoder_outputs = self.model_wrapper.model.model.encoder(
            inputs_embeds=modified_embeddings,
            attention_mask=attention_mask,
        )

        # Expand encoder outputs for multiple return sequences
        if num_samples > 1:
            encoder_hidden_states = encoder_outputs.last_hidden_state.repeat_interleave(
                num_samples, dim=0
            )
            encoder_attention_mask = (
                attention_mask.repeat_interleave(num_samples, dim=0)
                if attention_mask is not None
                else None
            )
        else:
            encoder_hidden_states = encoder_outputs.last_hidden_state
            encoder_attention_mask = attention_mask

        # Step 2: Initialize decoder inputs for inference
        decoder_start_token_id = self.model_wrapper.model.model.config.decoder_start_token_id
        decoder_input_ids = torch.full(
            (batch_size * num_samples, 1), 
            decoder_start_token_id, 
            dtype=torch.long, 
            device=device
        )

        # Step 3: Autoregressive generation with output injection
        generated_ids = decoder_input_ids
        past_key_values = None

        for step in range(max_length):  # -1 because we start with one token
            # Decoder forward pass
            decoder_outputs = self.model_wrapper.model.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Get hidden states for current step
            current_hidden_states = decoder_outputs.last_hidden_state[:, -1:, :] # last token

            # Pass through linear head to generate logits
            if self.model_wrapper.model.model.config.tie_word_embeddings:
                current_hidden_states = current_hidden_states * (self.model_wrapper.model.model.model_dim**-0.5)
            logits = self.model_wrapper.model.model.lm_head(current_hidden_states).squeeze(1)  # (batch, vocab_size)

            # Apply output injection if available
            if future_covariates is not None:
                # Expand future covariates for multiple return sequences
                if num_samples > 1:
                    future_covariates = future_covariates.repeat_interleave(num_samples, dim=0)

                # Get corresponding future covariate for current step
                # Assumes that only future covariates after the predicted tokens are used, so perpetually size 1
                current_future_cov = future_covariates[:, step : step + 1, :]  # (batch, 1, cov_dim)
                next_token_logits = self.covariate_adapter.inject_output_covariates(
                    current_hidden_states, current_future_cov, logits
                )
            else:
                next_token_logits = logits

            # Sample next token
            temperature = self.model_wrapper.model.config.temperature
            top_k = self.model_wrapper.model.config.top_k
            top_p = self.model_wrapper.model.config.top_p

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(1)], dim=1)
            past_key_values = decoder_outputs.past_key_values

        # Remove decoder start token
        generated_ids = generated_ids[:, 1:]

        generated_ids = generated_ids.reshape(input_ids.size(0), num_samples, -1)

        # Convert generated IDs back to tokens
        predictions = self.model_wrapper.tokenizer.output_transform(
            generated_ids.to("cpu"), scale.to("cpu")
        )

        return predictions

    def to(self, device: Union[torch.device, str], **kwargs):
        """Moves all model parameters and buffers to the specified device.
        
        Args:
            device: The destination device (can be a device object or a string like 'cuda' or 'cpu')
            **kwargs: Additional arguments to pass to the underlying model's to() method
            
        Returns:
            Model: self
        """
        # Convert device string to torch.device if needed
        if isinstance(device, str):
            device = torch.device(device)

        # Call parent's to() method to handle basic parameters and buffers
        super().to(device, **kwargs)

        # Move the wrapped model to the device
        if hasattr(self.model_wrapper, '_get_base_model'):
            base_model = self.model_wrapper._get_base_model()
            if hasattr(base_model, 'to'):
                base_model.to(device, **kwargs)
            # Handle special case for ChronosPipeline
            elif hasattr(base_model, 'model') and hasattr(base_model.model, 'to'):
                base_model.model.to(device, **kwargs)

        # Move the covariate adapter
        if hasattr(self, 'covariate_adapter'):
            self.covariate_adapter.to(device, **kwargs)

        return self

    def save_pretrained(self, save_directory: str):
        """Save the model (only adapter weights + config)"""
        os.makedirs(save_directory, exist_ok=True)

        # Save only the adapter weights
        adapter_state_dict = self.covariate_adapter.state_dict()
        torch.save(adapter_state_dict, os.path.join(save_directory, "adapter_weights.bin"))

        # Save configuration
        config = {
            "base_model_path": self._base_model_path,
            "model_type": self.model_type,
            "covariate_dim": self.covariate_dim,
            "hidden_dim": self.hidden_dim,
            "freeze_pretrained": self.freeze_pretrained,
            "d_model": self.model_wrapper.d_model,
            "output_dim": self.model_wrapper.output_dim
        }

        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"Universal covariate model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, save_directory: str, **kwargs):
        """Load a saved model"""
        # Load configuration
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Recreate the model using the appropriate loading function
        model = load_adapted_x_model(
            model_name_or_path=config["base_model_path"],
            model_type=config["model_type"],
            covariate_dim=config["covariate_dim"],
            hidden_dim=config["hidden_dim"],
            freeze_pretrained=config["freeze_pretrained"],
            **kwargs
        )

        # Load the saved adapter weights
        adapter_weights_path = os.path.join(save_directory, "adapter_weights.bin")
        if os.path.exists(adapter_weights_path):
            adapter_state_dict = torch.load(adapter_weights_path, map_location="cpu")
            model.covariate_adapter.load_state_dict(adapter_state_dict)
            print(f"Loaded adapter weights from {adapter_weights_path}")

        return model

# Universal loading function
def load_adapted_x_model(
    model_name_or_path: str,
    model_type: str,  # "chronos", "timesfm", "chronos_bolt", etc.
    covariate_dim: int,
    hidden_dim: int = 256,
    freeze_pretrained: bool = True,
    specific_model_config: Dict[str, Any] = {},
    **model_kwargs
) -> AdaptedXModel:
    """
    Universal function to load any time series model with covariate capability
    """
   
    if model_type.lower() == "chronos":
        base_model = load_chronos_model(model_name_or_path, **model_kwargs)
        model_wrapper = ChronosWrapper(base_model, specific_model_config)
    elif model_type.lower() == "chronos_bolt":
        base_model = load_chronos_bolt_model(model_name_or_path, **model_kwargs)
        model_wrapper = ChronosBoltWrapper(base_model)
    elif model_type.lower() == "timesfm":
        base_model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
        model_wrapper = TimesFMWrapper(base_model)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # print(base_model)
    
    # Create universal model
    universal_model = AdaptedXModel(
        model_wrapper=model_wrapper,
        covariate_dim=covariate_dim,
        hidden_dim=hidden_dim,
        freeze_pretrained=freeze_pretrained,
        **model_kwargs
    )
    
    # Store base model path
    universal_model._base_model_path = model_name_or_path
    
    logger.info(f"Loaded {model_type} model with covariate capability")
    logger.info(f"Model dimension: {model_wrapper.d_model}, Covariate dimension: {covariate_dim}")
    
    return universal_model


# Placeholder loading functions - replace with actual implementations
def load_chronos_model(model_name_or_path: str, **kwargs):
    try:
        from chronos import ChronosPipeline
        return ChronosPipeline.from_pretrained(model_name_or_path, **kwargs)
    except ImportError:
        raise ImportError("chronos library not installed. Install with: pip install chronos-forecasting")


def load_chronos_bolt_model(model_name_or_path: str, **kwargs):
    try:
        from chronos import ChronosBoltPipeline
        return ChronosBoltPipeline.from_pretrained(model_name_or_path, **kwargs)
    except ImportError:
        raise ImportError("chronos library not installed. Install with: pip install chronos-forecasting")


def load_timesfm_model(model_name_or_path: str, **kwargs):
    """Placeholder for TimesFM model loading (currently not in use)"""

    # from timesfm import TimesFM  # Hypothetical import
    # base_model = TimesFM.from_pretrained(model_name_or_path, **kwargs)

    # Replace with actual TimesFM loading logic
    class MockTimesFMModel:
        def __init__(self):
            self.config = type('Config', (), {'d_model': 512, 'prediction_length': 24})()
    
    return MockTimesFMModel()


# # Example usage
# def example_usage():
#     """Example showing universal usage"""

#     # Load different models with covariates
#     models = {}

#     # Chronos model
#     try:
#         models['chronos'] = load_adapted_x_model(
#             model_name_or_path="amazon/chronos-t5-tiny",
#             model_type="chronos",
#             covariate_dim=5
#         )
#         print("✅ Loaded Chronos model")
#     except Exception as e:
#         print(f"❌ Failed to load Chronos: {e}")

#     # ChronosBolt model
#     try:
#         models['chronos_bolt'] = load_adapted_x_model(
#             model_name_or_path="amazon/chronos-bolt-tiny",
#             model_type="chronos_bolt",
#             covariate_dim=5
#         )
#         print("✅ Loaded ChronosBolt model")
#     except Exception as e:
#         print(f"❌ Failed to load ChronosBolt: {e.with_traceback()}")

#     # TimesFM model
#     try:
#         models['timesfm'] = load_adapted_x_model(
#             model_name_or_path="google/timesfm-2.0-500m-pytorch",
#             model_type="timesfm",
#             covariate_dim=5
#         )
#         print("✅ Loaded TimesFM model")
#     except Exception as e:
#         print(f"❌ Failed to load TimesFM: {e}")

#     # Save and load
#     # for name, model in models.items():
#     #     save_path = f"./my_{name}_model"
#     #     model.save_pretrained(save_path)
#     #     loaded_model = AdaptedXModel.from_pretrained(save_path)
#     #     print(f"✅ Saved and loaded {name} model")

# if __name__ == "__main__":
#     example_usage()
