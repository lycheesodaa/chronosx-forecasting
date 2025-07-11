"""
Wrapper classes for different time series foundation models
Provides unified interface for ChronosX adapter integration
"""
import torch
import torch.nn as nn
from typing import Any, Tuple, Optional
from abc import ABC, abstractmethod

from transformers import GenerationConfig

class ModelWrapper(ABC):
    """Abstract base class for model wrappers"""
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type string"""
        pass
    
    @property
    @abstractmethod
    def d_model(self) -> int:
        """Return the model dimension"""
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the output dimension"""
        pass
        
    @abstractmethod
    def _get_base_model(self) -> Any:
        """Return the underlying model instance"""
        pass

    @abstractmethod
    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Pass to the underlying embed tokens function"""
        pass
    
    @abstractmethod
    def get_input_embeddings(self, input_data: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Any]:
        """
        Convert input data to embeddings
        
        Returns:
            embeddings: Tensor of shape (batch_size, seq_len, d_model)
            additional_info: Any additional information needed for forward pass
        """
        pass
    
    @abstractmethod
    def forward_with_embeddings(self, embeddings: torch.Tensor, additional_info: Any, 
                               **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with custom embeddings
        
        Returns:
            hidden_states: Hidden states from the model
            outputs: Final model outputs
        """
        pass

    @abstractmethod
    def post_processing(self, sequences: torch.Tensor, hidden_states: torch.Tensor, 
                        logits: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Performing any necessary processing e.g. converting logits back into outputs
        """
        pass


# ------------- Modify/add wrappers for models within this section -------------
class ChronosWrapper(ModelWrapper):
    """Wrapper for Chronos models (T5-based tokenized models)"""
    from chronos import ChronosPipeline, ChronosTokenizer, ChronosModel

    chronos_pipeline: ChronosPipeline
    tokenizer: ChronosTokenizer
    model: ChronosModel

    def __init__(self, chronos_pipeline, specific_model_config):
        self.chronos_pipeline = chronos_pipeline
        self.tokenizer = chronos_pipeline.tokenizer
        self.model = chronos_pipeline.model

        self.model.config.num_samples = specific_model_config.get("num_samples", 20)

    def _get_base_model(self):
        """Return the underlying Chronos pipeline instance"""
        return self.chronos_pipeline

    @property
    def model_type(self) -> str:
        return "chronos"

    @property
    def d_model(self) -> int:
        # Get d_model from the model config
        return self.model.model.config.d_model

    @property
    def output_dim(self) -> int:
        # For Chronos, output dimension is based on vocab size and prediction length
        config = self.model.config
        return config.prediction_length

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Pass to the underlying embed tokens function"""
        if hasattr(self.model.model, 'encoder'):
            return self.model.model.encoder.embed_tokens(token_ids)
        else:
            return self.model.model.embed_tokens(token_ids)

    def get_input_embeddings(self, input_data: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Any]:
        """Get input embeddings from Chronos tokenizer and model"""
        input_data = input_data.to(self.tokenizer.boundaries.device)
        embeddings, _ = self.chronos_pipeline.embed(context=input_data)
        embeddings = embeddings.to(self.model.device)

        return embeddings

    def forward_with_embeddings(self, input_data: torch.Tensor, embeddings: torch.Tensor, 
                               **kwargs) -> Tuple[torch.Tensor, Tuple[torch.Tensor], torch.Tensor]:
        """Forward pass with custom embeddings"""
        input_data = input_data.to(self.tokenizer.boundaries.device)
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(
            input_data
        )
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        self.scale = scale

        assert hasattr(self.model.model, "generate")

        sequences = self.model.model.generate(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            generation_config=GenerationConfig(
                min_new_tokens=self.model.config.prediction_length,
                max_new_tokens=self.model.config.prediction_length,
                do_sample=True,
                num_return_sequences=self.model.config.num_samples,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.pad_token_id,
                temperature=self.model.config.temperature,
                top_k=self.model.config.top_k,
                top_p=self.model.config.top_p,
                output_hidden_states=True,
                output_logits=True,
                return_dict_in_generate=True,
            ),
        )

        sequences = outputs.sequences
        hidden_states = outputs.decoder_hidden_states
        logits = outputs.logits

        if self.model.config.model_type == "seq2seq":
            sequences = sequences[..., 1:]  # remove the decoder start token
        else:
            assert self.model.config.model_type == "causal"
            assert sequences.size(-1) == input_ids.size(-1) + self.model.config.prediction_length
            sequences = sequences[..., -self.model.config.prediction_length:]

        sequences = sequences.reshape(input_ids.size(0), self.model.config.num_samples, -1)

        # TODO bother with the hidden states and logits when you need the OIB
        # hidden_states = hidden_states.reshape(input_ids.size(0), self.model.config.num_samples, -1)
        # logits = logits.reshape(input_ids.size(0), self.model.config.num_samples, -1)

        # print("sequences.shape", sequences.shape)
        # print("hidden_states.shape", len(hidden_states), len(hidden_states[0]), hidden_states[0][0].shape)
        # print("logits.shape", len(logits), logits[0].shape)

        return sequences, hidden_states, logits

    def post_processing(self, sequences: torch.Tensor, hidden_states: torch.Tensor, 
                        logits: torch.Tensor) -> torch.Tensor:
        """
        Performing any necessary processing e.g. converting logits back into outputs
        """
        output = self.tokenizer.output_transform(
            sequences.to(self.scale.device), self.scale
        )

        return output.to(self.model.device).median(dim=1)


class ChronosBoltWrapper(ModelWrapper):
    """Wrapper for ChronosBolt models"""
    
    def __init__(self, chronos_bolt_model):
        self.chronos_bolt_model = chronos_bolt_model
        
    def _get_base_model(self):
        """Return the underlying ChronosBolt model instance"""
        return self.chronos_bolt_model
        
    @property
    def model_type(self) -> str:
        return "chronos_bolt"
    
    @property
    def d_model(self) -> int:
        return self.chronos_bolt_model.config.d_model
    
    @property
    def output_dim(self) -> int:
        return (self.chronos_bolt_model.num_quantiles * 
               self.chronos_bolt_model.chronos_config.prediction_length)
    
    def get_input_embeddings(self, input_data: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Any]:
        """Get input embeddings from ChronosBolt encode method"""
        mask = kwargs.get("mask")
        
        # Use the model's encode method to get embeddings and other info
        encoder_outputs, loc_scale, input_embeds, attention_mask = self.chronos_bolt_model.encode(
            context=input_data, 
            mask=mask
        )
        
        additional_info = {
            "encoder_outputs": encoder_outputs,
            "loc_scale": loc_scale, 
            "attention_mask": attention_mask
        }
        
        return input_embeds, additional_info
    
    def forward_with_embeddings(self, embeddings: torch.Tensor, additional_info: dict, 
                               **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with custom embeddings"""
        encoder_outputs = additional_info["encoder_outputs"]
        loc_scale = additional_info["loc_scale"]
        attention_mask = additional_info["attention_mask"]
        
        # Use the model's decode method
        sequence_output = self.chronos_bolt_model.decode(
            input_embeds=embeddings,
            attention_mask=attention_mask,
            hidden_states=encoder_outputs
        )
        
        # Get quantile predictions using output patch embedding
        batch_size = embeddings.shape[0]
        quantile_preds_shape = (
            batch_size,
            self.chronos_bolt_model.num_quantiles,
            self.chronos_bolt_model.chronos_config.prediction_length,
        )
        
        quantile_preds = self.chronos_bolt_model.output_patch_embedding(sequence_output).view(
            *quantile_preds_shape
        )
        
        # Unscale predictions using instance normalization
        quantile_preds = self.chronos_bolt_model.instance_norm.inverse(
            quantile_preds.view(batch_size, -1),
            loc_scale,
        ).view(*quantile_preds_shape)
        
        return sequence_output, quantile_preds


class TimesFMWrapper(ModelWrapper):
    """Wrapper for TimesFM models"""
    
    def __init__(self, timesfm_model):
        self.timesfm_model = timesfm_model
        
    def _get_base_model(self):
        """Return the underlying TimesFM model instance"""
        return self.timesfm_model
        
    @property
    def model_type(self) -> str:
        return "timesfm"
    
    @property
    def d_model(self) -> int:
        # Get model dimension from config or fallback
        if hasattr(self.timesfm_model, 'config') and hasattr(self.timesfm_model.config, 'd_model'):
            return self.timesfm_model.config.d_model
        elif hasattr(self.timesfm_model, 'd_model'):
            return self.timesfm_model.d_model
        else:
            # Fallback - common dimension for TimesFM
            return 512
    
    @property
    def output_dim(self) -> int:
        # Get output dimension from config or fallback
        if hasattr(self.timesfm_model, 'config') and hasattr(self.timesfm_model.config, 'prediction_length'):
            return self.timesfm_model.config.prediction_length
        elif hasattr(self.timesfm_model, 'prediction_length'):
            return self.timesfm_model.prediction_length
        else:
            # Fallback - common prediction length
            return 24
    
    def get_input_embeddings(self, input_data: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Any]:
        """Get input embeddings from TimesFM model"""
        # TimesFM typically uses patching - check if model has a patch method
        if hasattr(self.timesfm_model, 'patch_embed') or hasattr(self.timesfm_model, 'patch'):
            # Use model's patching if available
            if hasattr(self.timesfm_model, 'patch_embed'):
                embeddings = self.timesfm_model.patch_embed(input_data)
            else:
                embeddings = self.timesfm_model.patch(input_data)
        else:
            # Fallback - create simple patching
            batch_size, seq_len = input_data.shape
            patch_size = getattr(self.timesfm_model.config, 'patch_size', 32) if hasattr(self.timesfm_model, 'config') else 32
            
            # Reshape to patches
            num_patches = seq_len // patch_size
            if num_patches * patch_size < seq_len:
                # Pad to make divisible by patch_size
                pad_length = num_patches * patch_size + patch_size - seq_len
                input_data = torch.nn.functional.pad(input_data, (0, pad_length))
                num_patches += 1
            
            patches = input_data[:, :num_patches * patch_size].view(batch_size, num_patches, patch_size)
            
            # Project to d_model dimension
            if not hasattr(self, 'fallback_projection'):
                self.fallback_projection = nn.Linear(patch_size, self.d_model).to(input_data.device)
            
            embeddings = self.fallback_projection(patches)
        
        additional_info = {
            "original_input": input_data,
            "input_shape": input_data.shape
        }
        
        return embeddings, additional_info
    
    def forward_with_embeddings(self, embeddings: torch.Tensor, additional_info: dict, 
                               **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with custom embeddings"""
        # TimesFM specific forward pass
        if hasattr(self.timesfm_model, 'transformer'):
            # Use transformer if available
            hidden_states = self.timesfm_model.transformer(embeddings)
        elif hasattr(self.timesfm_model, 'encoder'):
            # Use encoder if available
            hidden_states = self.timesfm_model.encoder(embeddings)
        else:
            # Fallback - embeddings pass through
            hidden_states = embeddings
        
        # Get predictions using prediction head
        if hasattr(self.timesfm_model, 'prediction_head'):
            outputs = self.timesfm_model.prediction_head(hidden_states)
        elif hasattr(self.timesfm_model, 'head'):
            outputs = self.timesfm_model.head(hidden_states)
        else:
            # Fallback prediction head
            if not hasattr(self, 'fallback_head'):
                self.fallback_head = nn.Linear(self.d_model, self.output_dim).to(embeddings.device)
            outputs = self.fallback_head(hidden_states.mean(dim=1))  # Global average pooling
        
        return hidden_states, outputs
