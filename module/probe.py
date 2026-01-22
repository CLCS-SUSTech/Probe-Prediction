import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig, GradientCheckpointingLayer
from transformers.modeling_outputs import SequenceClassifierOutputWithPast, BaseModelOutputWithPast
from transformers.utils import can_return_tuple
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from typing import Optional
import logging
from abc import ABC, abstractmethod
from typing import Type, Dict, Tuple, Union, Callable
import json
import os
from peft import get_peft_model, LoraConfig, TaskType
from peft.utils import set_peft_model_state_dict

logger = logging.getLogger(__name__)

def get_probe_types():
    return {
        "multi_layer": MultiLayerProbe,
        "extra_layer": ExtraLayerProbe,
        "lora": LoraProbe,
        "linear": LinearProbe
    }

class BaseModelComponents(ABC):

    def get_decoder_layer(self) -> Type[nn.Module]:
        '''
        Note: each decoder layer class should be initialized with arguments (config, layer_idx).

        Returns:
            decoder_layer_cls: The decoder layer class of the base model.
        '''
        raise NotImplementedError

    def get_position_embedding_info(self) -> Type[nn.Module]:
        '''
        Note: each embedding class should be initialized with arguments (config).

        Returns:
            pe_cls: The position embedding class of the base model.
        
        Example:
            return Olmo2RotaryEmbedding        
        '''
        raise NotImplementedError

    def get_norm_info(self) -> Tuple[Type[nn.Module], Dict[str, str]]:
        '''
        Returns:
            norm_cls, param_map: A tuple of (normalization class, the map of keys in probe config for normalization parameters) used in the base model.
        
        Example:
            return (Olmo2RMSNorm, {
            'hidden_size' : 'hidden_size',
            'eps': 'rms_norm_eps'
            })
        '''
        raise NotImplementedError


class ProbeConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_json_file(cls, json_file: str, **kwargs) -> PretrainedConfig:
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict, **kwargs)

class ProbeWrapper(PreTrainedModel):
    config_class = ProbeConfig
    pretrained_model_prefix = "model"

    _supports_flash_attn_3 = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_attention_backend = True
    supports_gradient_checkpointing = True
    
    def __init__(self, config: ProbeConfig, pretrained_model: PreTrainedModel, pretrained_model_components: BaseModelComponents):
        
        for attr in pretrained_model.config.__dict__.keys():
            if not hasattr(config, attr):
                setattr(config, attr, getattr(pretrained_model.config, attr))

        if hasattr(pretrained_model.config, "_attn_implementation"):
             setattr(config, "_attn_implementation", pretrained_model.config._attn_implementation)
        else:
             setattr(config, "_attn_implementation", "eager")
        
        super().__init__(config)
        self.pretrained_model = pretrained_model
        self.pretrained_model.requires_grad_(False)

        self.pretrained_model_components = pretrained_model_components
        self.loss_fn = nn.MSELoss()

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutputWithPast:
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.pretrained_model.get_input_embeddings()(input_ids)

        cache_position = torch.arange(
            0, inputs_embeds.shape[1], device=inputs_embeds.device
        )

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        transformer_outputs : BaseModelOutputWithPast = self.pretrained_model(
            input_ids=None, # pass inputs_embeds instead
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache, # we only involve inference, hence use_cache is not used in practice
            output_attentions=output_attentions,
            output_hidden_states=True,
            cache_position=cache_position,
            **kwargs
        )

        logits = self.probe_forward(
            transformer_outputs=transformer_outputs,
            position_ids=position_ids,
            causal_mask=causal_mask,
            cache_position=cache_position,
            **kwargs
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.pretrained_model.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.pretrained_model.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.pretrained_model.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        pooled_logits = pooled_logits.flatten()

        loss = None
        if labels is not None:
            labels = labels.flatten()
            if labels.dtype != torch.float32:
                logger.warning_once(
                    f'MSELoss requires labels in torch.float32, but got {labels.dtype}. Converting labels to float32.'
                )
                labels = labels.to(torch.float32)
            if pooled_logits.dtype != torch.float32:
                pooled_logits = pooled_logits.to(torch.float32)
            loss = self.loss_fn(pooled_logits, labels)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
        )
    
    @abstractmethod
    def probe_forward(self, transformer_outputs: BaseModelOutputWithPast, *args, **kwargs) -> torch.Tensor:
        pass

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        if is_main_process:
            self.config.save_pretrained(save_directory)

        self.save_pretrained_probe(save_directory, safe_serialization=safe_serialization, **kwargs)
    
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: str, 
        pretrained_model: Optional[PreTrainedModel] = None, # It is infact required to pass in the base model
        pretrained_model_components: Optional[BaseModelComponents] = None, # It is infact required to pass in the base model components
        **kwargs
    ):
        if pretrained_model is None or pretrained_model_components is None:
             raise ValueError("`from_pretrained` for ProbeWrapper requires external `pretrained_model` and `pretrained_model_components` instances.")
        config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        probe_types = {
            "multi_layer": MultiLayerProbe,
            "extra_layer": ExtraLayerProbe,
            "lora": LoraProbe,
        }
        if not hasattr(config, "probe_type"):
            config.probe_type = "multi_layer"
            logger.warning_once(f"`probe_type` not found in the config file. Defaulting to 'MultiLayer' probe.")
        probe_cls = probe_types.get(config.probe_type, MultiLayerProbe)
        model = probe_cls(config, pretrained_model, pretrained_model_components)
        model.load_pretrained_probe(pretrained_model_name_or_path, **kwargs)

        return model

    def save_pretrained_probe(self, save_directory: str, safe_serialization: bool = True, **kwargs):
        
        full_state_dict = self.state_dict()
        probe_state_dict = {}
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name in full_state_dict:
                    probe_state_dict[name] = full_state_dict[name].cpu()

        os.makedirs(save_directory, exist_ok=True)
        if safe_serialization:
            from safetensors.torch import save_file
            output_path = os.path.join(save_directory, "probe.safetensors")
            save_file(probe_state_dict, output_path, metadata={"format": "safetensors"})
        else:
            output_path = os.path.join(save_directory, "probe.bin")
            torch.save(probe_state_dict, output_path)
            
        logger.info(f"Saved {len(probe_state_dict)} tensors to {output_path}")
    
    def load_pretrained_probe(self, load_directory: str, **kwargs):
        safetensors_path = os.path.join(load_directory, "probe.safetensors")
        bin_path = os.path.join(load_directory, "probe.bin")

        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
            load_path = safetensors_path
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path)
            load_path = bin_path
        else:
            raise FileNotFoundError(f"Probe weights file not found in {load_directory}. Expected 'probe.safetensors' or 'probe.bin'.")
        
        loaded_keys = {}
        for k, v in state_dict.items():
            clean_k = k.replace("module.", "")
            loaded_keys[clean_k] = v

        loaded_count = 0
        missing_trainable_params = []

        for name, param in self.named_parameters():
            clean_name = name.replace("module.", "")
            
            if clean_name in loaded_keys:
                loaded_tensor = loaded_keys[clean_name]
                param.data.copy_(loaded_tensor.to(device=param.device, dtype=param.dtype))
                loaded_count += 1
            else:
                if param.requires_grad:
                    missing_trainable_params.append(name)

        if missing_trainable_params:
            logger.warning("="*60)
            logger.warning(f"!!! WARNING: {len(missing_trainable_params)} trainable parameters were NOT loaded !!!")
            logger.warning(f"Missing keys: {missing_trainable_params}")
            logger.warning("This might lead to random performance if not intended.")
            logger.warning("="*60)
        else:
            logger.info(f"Successfully loaded all {loaded_count} trainable parameters.")
    
    
class ExtraLayerProbe(ProbeWrapper):

    def __init__(self, config: ProbeConfig, pretrained_model: PreTrainedModel, pretrained_model_components: BaseModelComponents):
        super().__init__(config, pretrained_model, pretrained_model_components)
        self.proj = nn.Linear(self.pretrained_model.config.hidden_size, self.config.hidden_size) \
            if self.config.hidden_size != self.pretrained_model.config.hidden_size else nn.Identity()
        self.probe = self.pretrained_model_components.get_decoder_layer()(self.config, layer_idx=self.pretrained_model.config.num_hidden_layers)
        norm_class, norm_config_map = self.pretrained_model_components.get_norm_info()
        norm_kwargs = {}
        for norm_param_name, config_key in norm_config_map.items():
            norm_kwargs[norm_param_name] = getattr(
                self.config,
                config_key, 
                getattr(self.pretrained_model.config, config_key, None)
            )
        self.norm = norm_class(**norm_kwargs)
        self.pe = self.pretrained_model_components.get_position_embedding_info()(self.config)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

        # explicitly cast to the same dtype as the pretrained model
        target_dtype = next(self.pretrained_model.parameters()).dtype
        self.to(target_dtype)
        
    def probe_forward(self, transformer_outputs: BaseModelOutputWithPast, position_ids, causal_mask, cache_position, *args, **kwargs) -> torch.Tensor:
        hidden_states = transformer_outputs.hidden_states[-1]
        hidden_states = self.proj(hidden_states)

        position_embeddings = self.pe(hidden_states, position_ids)
        
        hidden_states = self.probe(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs
        )

        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        hidden_states = self.norm(hidden_states)
        logits = self.classifier(hidden_states)
        return logits

class MultiLayerProbe(ProbeWrapper):

    def __init__(self, config: ProbeConfig, pretrained_model: PreTrainedModel, pretrained_model_components: BaseModelComponents):
        super().__init__(config, pretrained_model, pretrained_model_components)

        self.projs = nn.ModuleList(
            [
                nn.Linear(self.pretrained_model.config.hidden_size, self.config.hidden_size) \
                if self.config.hidden_size != self.pretrained_model.config.hidden_size else nn.Identity()
                for _ in range(self.config.num_hidden_layers)
            ]
        )

        self.probes = nn.ModuleList(
            [
                self.pretrained_model_components.get_decoder_layer()(self.config, layer_idx=i)
                for i in range(self.config.num_hidden_layers)
            ]
        )

        norm_class, norm_config_map = self.pretrained_model_components.get_norm_info()

        norm_kwargs = {}
        for norm_param_name, config_key in norm_config_map.items():
            norm_kwargs[norm_param_name] = getattr(
                self.config,
                config_key, 
                getattr(self.pretrained_model.config, config_key, None)
            )

        self.norms = nn.ModuleList([
            norm_class(**norm_kwargs) for _ in range(self.config.num_hidden_layers + 1)
        ])
        
        self.pe = self.pretrained_model_components.get_position_embedding_info()(self.config)
        
        self.layer_map_base2probe = self.config.layer_map_base2probe # array of int

        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

        # explicitly cast to the same dtype as the pretrained model
        target_dtype = next(self.pretrained_model.parameters()).dtype
        self.to(target_dtype)

    def probe_forward(self, transformer_outputs: BaseModelOutputWithPast, position_ids, causal_mask, cache_position, *args, **kwargs) -> torch.Tensor:
        all_hidden_states = transformer_outputs.hidden_states

        position_embeddings = self.pe(all_hidden_states[0], position_ids)
        
        for probe_idx, base_layer_idx in enumerate(self.layer_map_base2probe):
            if probe_idx == 0:
                hidden_states = self.norms[probe_idx](self.projs[probe_idx](all_hidden_states[base_layer_idx]))
            else:
                hidden_states = hidden_states + self.norms[probe_idx](self.projs[probe_idx](all_hidden_states[base_layer_idx]))
            
            hidden_states = self.probes[probe_idx](
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs
            )

            # check is tuple for version compatibility
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        
        hidden_states = self.norms[-1](hidden_states)
        logits = self.classifier(hidden_states)
        return logits

class LoraProbe(ProbeWrapper):

    def __init__(self, config: ProbeConfig, pretrained_model: PreTrainedModel, pretrained_model_components: BaseModelComponents):
        super().__init__(config, pretrained_model, pretrained_model_components)
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=getattr(config, "lora_rank", 8),
            lora_alpha=getattr(config, "lora_alpha", 32),
            lora_dropout=getattr(config, "lora_dropout", 0.1),
            target_modules=getattr(config, "lora_target_modules", None)
        )
        
        self.pretrained_model = get_peft_model(self.pretrained_model, lora_config)
        self.pretrained_model.print_trainable_parameters()
        
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        
        target_dtype = next(self.pretrained_model.parameters()).dtype
        self.classifier.to(target_dtype)

    def probe_forward(self, transformer_outputs: BaseModelOutputWithPast, *args, **kwargs) -> torch.Tensor:
        hidden_states = transformer_outputs.hidden_states[-1]
        logits = self.classifier(hidden_states)
        return logits
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutputWithPast:
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.pretrained_model.get_input_embeddings()(input_ids)

        cache_position = torch.arange(
            0, inputs_embeds.shape[1], device=inputs_embeds.device
        )

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # In LoraProbe, pretrained_model includes the lora adapter
        transformer_outputs = self.pretrained_model(
            input_ids=None, # pass inputs_embeds
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            cache_position=cache_position,
            **kwargs
        )

        logits = self.probe_forward(
            transformer_outputs=transformer_outputs,
            position_ids=position_ids,
            causal_mask=causal_mask,
            cache_position=cache_position,
            **kwargs
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        pooled_logits = pooled_logits.flatten()

        loss = None
        if labels is not None:
            labels = labels.flatten()
            if labels.dtype != torch.float32:
                logger.warning_once(
                    f'MSELoss requires labels in torch.float32, but got {labels.dtype}. Converting labels to float32.'
                )
                labels = labels.to(torch.float32)
            if pooled_logits.dtype != torch.float32:
                pooled_logits = pooled_logits.to(torch.float32)
            loss = self.loss_fn(pooled_logits, labels)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
        )
    
    def save_pretrained_probe(self, save_directory: str, safe_serialization: bool = True, **kwargs):
        self.pretrained_model.save_pretrained(save_directory)
        
        state_dict = {}
        state_dict["classifier.weight"] = self.classifier.weight
        state_dict["classifier.bias"] = self.classifier.bias
        
        if safe_serialization:
            from safetensors.torch import save_file
            output_path = os.path.join(save_directory, "classifier.safetensors")
            save_file(state_dict, output_path, metadata={"format": "safetensors"})
        else:
            output_path = os.path.join(save_directory, "classifier.bin")
            torch.save(state_dict, output_path)

        logger.info(f"Probe weights saved to {output_path}")
    
    def load_pretrained_probe(self, load_directory: str, **kwargs):
        # Load adapter
        safetensors_path = os.path.join(load_directory, "adapter_model.safetensors")
        bin_path = os.path.join(load_directory, "adapter_model.bin")
        
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            adapter_state_dict = load_file(safetensors_path)
            set_peft_model_state_dict(self.pretrained_model, adapter_state_dict)
        elif os.path.exists(bin_path):
            adapter_state_dict = torch.load(bin_path)
            set_peft_model_state_dict(self.pretrained_model, adapter_state_dict)
        else:
             logger.warning(f"No LoRA adapter found at {load_directory}")

        # Load classifier
        c_safetensors = os.path.join(load_directory, "classifier.safetensors")
        c_bin = os.path.join(load_directory, "classifier.bin")

        if os.path.exists(c_safetensors):
            from safetensors.torch import load_file
            state_dict = load_file(c_safetensors)
            load_path = c_safetensors
        elif os.path.exists(c_bin):
            state_dict = torch.load(c_bin)
            load_path = c_bin
        else:
            raise FileNotFoundError(f"Classifier weights not found in {load_directory}. Expected 'classifier.safetensors' or 'classifier.bin'.")

        classifier_state_dict = {
            "weight": state_dict["classifier.weight"],
            "bias": state_dict["classifier.bias"]
        }

        self.classifier.load_state_dict(classifier_state_dict)
        
        logger.info(f"Probe weights loaded from {load_path}")
    
class LinearProbe(ProbeWrapper):

    def __init__(self, config: ProbeConfig, pretrained_model: PreTrainedModel, pretrained_model_components: BaseModelComponents):
        super().__init__(config, pretrained_model, pretrained_model_components)
        
        self.projs = nn.ModuleList(
            [
                nn.Linear(self.pretrained_model.config.hidden_size, self.config.num_labels) \
                for _ in range(self.config.num_hidden_layers)
            ]
        )
        target_dtype = next(self.pretrained_model.parameters()).dtype
        self.to(target_dtype)

    def probe_forward(self, transformer_outputs, *args, **kwargs) -> torch.Tensor:
        all_hidden_states = transformer_outputs.hidden_states
        
        total_layer_logits = 0
        for layer_idx in range(self.config.num_hidden_layers):
            hidden_states = all_hidden_states[layer_idx]
            layer_logits = self.projs[layer_idx](hidden_states)
            total_layer_logits += layer_logits
        
        avg_layer_logits = total_layer_logits / self.config.num_hidden_layers
        cumsum_logits = torch.cumsum(avg_layer_logits, dim=1)
        
        seq_len = avg_layer_logits.size(1)
        device = avg_layer_logits.device
        time_steps = torch.arange(1, seq_len + 1, device=device).view(1, -1, 1)
        cumulative_mean_logits = cumsum_logits / time_steps
        
        return cumulative_mean_logits

    