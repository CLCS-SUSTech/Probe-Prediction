from module.probe import BaseModelComponents
from module.component import register_component

from transformers.models.olmo3.modeling_olmo3 import (
    Olmo3DecoderLayer,
    Olmo3RMSNorm,
    Olmo3RotaryEmbedding
)

@register_component('olmo3')
class Olmo3ModelComponents(BaseModelComponents):

    def get_decoder_layer(self):
        return Olmo3DecoderLayer
    
    def get_norm_info(self):
        return Olmo3RMSNorm, {
            'hidden_size' : 'hidden_size',
            'eps': 'rms_norm_eps'
            }
    
    def get_position_embedding_info(self):
        return Olmo3RotaryEmbedding
    