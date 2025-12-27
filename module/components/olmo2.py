from module.probe import BaseModelComponents
from module.component import register_component

from transformers.models.olmo2.modeling_olmo2 import (
    Olmo2DecoderLayer,
    Olmo2RMSNorm,
    Olmo2RotaryEmbedding
)

@register_component('olmo2')
class Olmo2ModelComponents(BaseModelComponents):

    def get_decoder_layer(self):
        return Olmo2DecoderLayer
    
    def get_norm_info(self):
        return Olmo2RMSNorm, {
            'hidden_size' : 'hidden_size',
            'eps': 'rms_norm_eps'
            }
    
    def get_position_embedding_info(self):
        return Olmo2RotaryEmbedding
    