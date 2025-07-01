import torch
from chronos.chronos import MeanScaleUniformBins

class DeviceAwareMeanScaleUniformBins(MeanScaleUniformBins):
    def _input_transform(self, context, scale=None):
        # Call parent's _input_transform but ensure boundaries are on the same device as context
        if scale is None:
            scale = self._get_scale(context)
        
        # Move scale to the same device as context
        if scale.device != context.device:
            scale = scale.to(context.device)
            
        # Scale the context
        scaled_context = context / scale.unsqueeze(dim=-1)
        
        # Move boundaries to the same device as scaled_context
        boundaries = self.boundaries.to(scaled_context.device)
        
        # Apply bucketization
        token_ids = (
            torch.bucketize(
                input=scaled_context,
                boundaries=boundaries,
                right=True,  # buckets are open to the right
            )
            + self.config.n_special_tokens
        )
        
        token_ids.clamp_(0, self.config.n_tokens - 1)
        
        # Create attention mask (1 for non-padding, 0 for padding)
        attention_mask = torch.ones_like(token_ids, dtype=torch.bool)
        
        return token_ids, attention_mask, scale
