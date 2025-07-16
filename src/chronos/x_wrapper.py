"""
Wrapper classes for different time series foundation models
Provides unified interface for ChronosX adapter integration
"""
import torch
import torch.nn as nn
from typing import Any, Tuple, Optional
from abc import ABC, abstractmethod

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
    def input_transform(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pass to the underlying embed tokens function"""
        pass
    
    @abstractmethod
    def get_input_embeddings(self, input_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Convert input data to embeddings
        
        Returns:
            embeddings: Tensor of shape (batch_size, seq_len, d_model)
        """
        pass
    
    @abstractmethod
    def forward_with_embeddings(self, embeddings: torch.Tensor, 
                               **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with custom embeddings
        
        Returns:
            hidden_states: Hidden states from the model
            outputs: Final model outputs
        """
        pass

    @abstractmethod
    def post_processing(self, hidden_states: torch.Tensor, 
                        logits: torch.Tensor) -> torch.Tensor:
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

        # override default chronos config arguments here
        self.model.config.num_samples = specific_model_config.get("num_samples", 20)
        self.model.config.prediction_length = specific_model_config.get("prediction_length", 64)

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

    def input_transform(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pass to the underlying input_transform function"""
        context = context.to(self.tokenizer.boundaries.device)
        context_tensor = self.chronos_pipeline._prepare_and_validate_context(context=context)
        token_ids, attention_mask, scale = self.tokenizer.context_input_transform(context_tensor)
        token_ids = token_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        scale = scale.to(self.model.device)
        return token_ids, attention_mask, scale

    def get_input_embeddings(self, input_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get input embeddings from Chronos tokenizer and model"""
        input_data = input_data.to(self.tokenizer.boundaries.device)
        embeddings, _ = self.chronos_pipeline.embed(context=input_data)
        embeddings = embeddings.to(self.model.device)

        return embeddings

    def forward_with_embeddings(self, input_data: torch.Tensor, 
                                labels: torch.Tensor,
                                attention_mask: torch.Tensor,
                                embeddings: torch.Tensor,
                                decoder_input_ids: Optional[torch.Tensor] = None,
                                decoder_attention_mask: Optional[torch.Tensor] = None,
                                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with custom embeddings"""
        encoder_outputs = self._run_encoder_with_embeddings(
            embeddings=embeddings,
            attention_mask=attention_mask,
        )

        decoder_outputs = self.model.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
        )

        logits = self.post_processing(decoder_outputs.last_hidden_state)

        return decoder_outputs.last_hidden_state, logits

    def _run_encoder_with_embeddings(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Run T5 encoder with custom embeddings instead of token IDs
        """
        encoder = self.model.model.encoder

        # Apply dropout to embeddings
        embeddings = encoder.dropout(embeddings)

        # Run through encoder stack
        encoder_outputs = encoder(
            inputs_embeds=embeddings,  # Use embeddings instead of input_ids
            attention_mask=attention_mask,
        )

        return encoder_outputs

    def post_processing(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Performing any necessary processing e.g. converting logits back into outputs
        """
        # Generate logits
        if self.model.model.config.tie_word_embeddings:
            hidden_states = hidden_states * (
                self.model.model.model_dim**-0.5
            )

        logits = self.model.model.lm_head(hidden_states).squeeze(1)  # (batch, vocab_size)

        return logits


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
