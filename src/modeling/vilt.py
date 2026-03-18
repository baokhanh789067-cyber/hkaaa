import copy
import os
import sys
import logging
import itertools
import pdb
import time
import math
from PIL import Image
from typing import List, Dict
from typing_extensions import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ViltConfig, ViltProcessor, ViltModel
from transformers import BertTokenizerFast

from src.modeling.continual_learner import EncoderWrapper, ContinualLearner
from src.modeling.adaptered_output import Adaptered_ViltOutput

# ================= [最终版] 双流 LoRA 定义 (FedDAT专用) =================
class LoRALayer(nn.Module):
    def __init__(self, linear_layer, rank=8, alpha=16):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.active_adapter = "adapter_0" # 默认使用本地路径

        # === 定义两套参数 ===
        # Set 0: Local (Personalized) -> 对应 adapter_0
        self.lora_A_0 = nn.Parameter(torch.zeros(rank, linear_layer.in_features))
        self.lora_B_0 = nn.Parameter(torch.zeros(linear_layer.out_features, rank))
        
        # Set 1: Global (Shared) -> 对应 adapter_1
        self.lora_A_1 = nn.Parameter(torch.zeros(rank, linear_layer.in_features))
        self.lora_B_1 = nn.Parameter(torch.zeros(linear_layer.out_features, rank))
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A_0, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_0)
        nn.init.kaiming_uniform_(self.lora_A_1, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_1)

    def forward(self, x):
        base_out = self.linear(x)
        
        # 路由逻辑
        if "adapter_1" in self.active_adapter: 
            # Global path
            lora_out = (x @ self.lora_A_1.T @ self.lora_B_1.T) * self.scaling
        elif "adapter_0" in self.active_adapter: 
            # Local path
            lora_out = (x @ self.lora_A_0.T @ self.lora_B_0.T) * self.scaling
        else:
            # Fallback (usually local)
            lora_out = (x @ self.lora_A_0.T @ self.lora_B_0.T) * self.scaling
            
        return base_out + lora_out
# =========================================================================

class ViltEncoderWrapper(EncoderWrapper):
    def __init__(self, processor: ViltProcessor, vilt: ViltModel, device: torch.device):
        super().__init__()
        self.processor = processor
        self.vilt = vilt
        self.device = device
        
        # 尝试加载本地 BERT tokenizer，失败则联网加载
        try:
            self.processor.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', local_files_only=True)
        except:
            self.processor.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        self.max_text_length = self.vilt.config.max_position_embeddings
        self.encoder_dim = self.vilt.config.hidden_size
        self.expand_modality_type_embeddings()

    def reset_processor(self, max_text_length: int, img_size: tuple):
        self.max_text_length = max_text_length
        self.processor.feature_extractor.size = img_size

    def process_inputs(self, images: List, texts: List[str]) -> Dict:
        encodings = self.processor(images=images, text=texts, max_length=self.max_text_length,
                                   padding=True, truncation=True, return_tensors="pt").to(self.device)
        return encodings

    def expand_modality_type_embeddings(self, type_vocab_size=3):
        self.vilt.config.modality_type_vocab_size = type_vocab_size
        emb_data = self.vilt.embeddings.token_type_embeddings.weight.data
        self.vilt.embeddings.token_type_embeddings = nn.Embedding(type_vocab_size, self.encoder_dim)
        self.vilt.embeddings.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
        self.vilt.embeddings.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
        self.vilt.embeddings.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

    def forward(self, **encodings: Dict) -> torch.FloatTensor:
        output = self.vilt(**encodings)
        return output.pooler_output

    def freeze_all_weights(self):
        for p in self.vilt.parameters():
            p.requires_grad = False

    def freeze_bottom_k_layers(self, k: int):
        assert k < len(self.vilt.encoder.layer)
        for p in self.vilt.embeddings.parameters():
            p.requires_grad = False
        for i in range(k):
            for p in self.vilt.encoder.layer[i].parameters():
                p.requires_grad = False


class ViltContinualLearner(ContinualLearner):
    def __init__(self, ordered_cl_tasks: List[str], encoder: ViltEncoderWrapper, encoder_dim: int, task_configs: Dict, device: torch.device, adapter_config):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.vilt_encoder = encoder # 关键：这里正确赋值了 vilt_encoder
        self.ordered_cl_tasks = ordered_cl_tasks
        self.task_configs = task_configs
        self.device = device
        self.adapter_config = adapter_config

        self.task_layer_dict = {}
        for task_key in ordered_cl_tasks:
            self.add_task_layer(task_key, task_configs[task_key])
        self.task_layer = nn.ModuleDict(self.task_layer_dict)

    def add_task_layer(self, task_key: str, task_config: Dict):
        num_labels = task_config["num_labels"]
        if task_config["model_type"] == "classification":
            num_images = task_config["num_images"]
            clf_layer = nn.Sequential(
                OrderedDict([
                    ("clf_fc0", nn.Linear(self.encoder_dim * num_images, self.encoder_dim * 2)),
                    ("clf_norm0", nn.LayerNorm(self.encoder_dim * 2)),
                    ("clf_actv0", nn.GELU()),
                    ("clf_fc1", nn.Linear(self.encoder_dim * 2, num_labels))
                ])
            )
            self.task_layer_dict[task_key] = clf_layer
        elif task_config["model_type"] == "multi-choice":
            clf_layer = nn.Sequential(
                OrderedDict([
                    ("clf_dropout", nn.Dropout(0.1)),
                    ("clf_fc0", nn.Linear(self.encoder_dim, 1)),
                ])
            )
            self.task_layer_dict[task_key] = clf_layer

    def forward(self, task_key: str, images: List, texts: List[str]):
        task_config = self.task_configs[task_key]
        if task_config['model_type'] == 'multi-choice':
            return self.forward_multi_choice(task_key, images, texts, task_config['num_choices'])
        elif task_config['model_type'] == 'classification':
            if task_config['num_images'] == 1:
                return self.forward_single_image(task_key, images, texts)
            else:
                return self.forward_multi_images(task_key, images, texts, task_config['num_images'])

    def forward_single_image(self, task_key: str, images: List, texts: List[str]):
        encodings = self.vilt_encoder.process_inputs(images, texts)
        encoder_output = self.vilt_encoder(**encodings)
        output_logits = self.task_layer[task_key](encoder_output)
        return encoder_output, output_logits

    def forward_multi_images(self, task_key: str, images: List[List], texts: List[str], num_images=2):
        flat_images_list = list(itertools.chain(*images))
        encodings = self.vilt_encoder.process_inputs(flat_images_list, texts)
        input_ids = encodings['input_ids']
        bs = len(input_ids)
        pixel_values = encodings['pixel_values'].view(bs, num_images, *encodings["pixel_values"].shape[-3:])
        pixel_mask = encodings['pixel_mask'].view(bs, num_images, *encodings["pixel_mask"].shape[-2:])

        pooler_outputs = []
        for i in range(num_images):
            encodings_i = {
                'input_ids': input_ids,
                'attention_mask': encodings['attention_mask'],
                'token_type_ids': encodings['token_type_ids'],
                'pixel_values': pixel_values[:, i, :, :, :],
                'pixel_mask': pixel_mask[:, i, :, :],
                'image_token_type_idx': i + 1,
            }
            pooled_out = self.vilt_encoder(**encodings_i)
            pooler_outputs.append(pooled_out)
        pooled_output = torch.cat(pooler_outputs, dim=-1)
        output_logits = self.task_layer[task_key](pooled_output)
        return pooled_output, output_logits

    def forward_multi_choice(self, task_key: str, images: List, texts: List[List[str]], num_choices):
        texts_list = list(itertools.chain(*texts))
        encodings = self.vilt_encoder.process_inputs(images, texts_list)
        bs = len(images)
        unflat_input_ids = encodings['input_ids'].view(bs, num_choices, -1)
        unflat_attention_mask = encodings['attention_mask'].view(bs, num_choices, -1)
        unflat_token_type_ids = encodings['token_type_ids'].view(bs, num_choices, -1)
        
        pooler_outputs = []
        for i in range(num_choices):
            encodings_i = {
                'input_ids': unflat_input_ids[:, i, :],
                'attention_mask': unflat_attention_mask[:, i, :],
                'token_type_ids': unflat_token_type_ids[:, i, :],
                'pixel_values': encodings['pixel_values'],
                'pixel_mask': encodings['pixel_mask']
            }
            pooled_out = self.vilt_encoder(**encodings_i)
            pooler_outputs.append(pooled_out)
        pooled_output = torch.stack(pooler_outputs, dim=0).transpose(0, 1)
        output_logits = self.task_layer[task_key](pooled_output).squeeze()
        return pooled_output, output_logits

    # ================= [关键方法] LoRA 注入与路由 (FedDAT) =================
    def add_lora(self, rank=8, alpha=16):
        """注入双流 LoRA 到 Attention 层"""
        for i in range(12):
            # 注意：ViLT 的 attention 结构通常是 .attention.attention
            attention = self.vilt_encoder.vilt.encoder.layer[i].attention.attention
            
            # 注入 Query
            if not isinstance(attention.query, LoRALayer):
                attention.query = LoRALayer(attention.query, rank, alpha)
                
            # 注入 Value
            if not isinstance(attention.value, LoRALayer):
                attention.value = LoRALayer(attention.value, rank, alpha)

    def set_active_adapter(self, adapter_name):
        """切换 LoRA 的路由 (Local/Global)"""
        for module in self.modules():
            if isinstance(module, LoRALayer):
                module.active_adapter = adapter_name
    
    # 兼容性接口 (防止 task_trainer 报错)
    def activate_gating(self): pass
    def deactivate_gating(self): pass
    def add_adapter(self): pass 
    # =================================================================

def load_vilt_encoder(logger, checkpoint_name: str, device: torch.device, pretrained_vilt_name: str) -> ViltEncoderWrapper:
    logger.info("-" * 100)
    logger.info("Loading ViLT encoder model: {}".format(checkpoint_name))
    vilt_processor = ViltProcessor.from_pretrained(pretrained_vilt_name)

    if checkpoint_name == pretrained_vilt_name:
        vilt = ViltModel.from_pretrained(pretrained_vilt_name)
        vilt_encoder = ViltEncoderWrapper(vilt_processor, vilt, device)
    else:
        config = ViltConfig.from_pretrained(pretrained_vilt_name)
        vilt = ViltModel(config)
        vilt_encoder = ViltEncoderWrapper(vilt_processor, vilt, device)
        if "nlvr2" in checkpoint_name:
            vilt_encoder.expand_modality_type_embeddings()
        ckpt = torch.load(checkpoint_name)
        vilt_encoder.load_state_dict(ckpt)

    logger.info("Successfully loaded pretrained ViLT encoder")
    return vilt_encoder

def create_vilt_continual_learner_model(logger, model_name_or_path: str,
                                        ordered_cl_tasks: List[str],
                                        model_config: Dict,
                                        task_configs: Dict,
                                        device: torch.device,):
    encoder = load_vilt_encoder(logger, checkpoint_name=model_name_or_path,
                                device=device,
                                pretrained_vilt_name=model_name_or_path)

    cl_model = ViltContinualLearner(ordered_cl_tasks=ordered_cl_tasks,
                                    encoder=encoder,
                                    encoder_dim=model_config["encoder_dim"],
                                    task_configs=task_configs,
                                    device=device,
                                    adapter_config=model_config['adapter_config'] if 'adapter_config' in model_config else None)
    logger.info("Successfully created and initialized ViLT Continual Leaner model")
    return cl_model

def convert_batch_to_vilt_input_dict(batch: Dict):
    return {"images": batch["images"], "texts": batch["raw_texts"]}

def convert_seq_batch_to_vilt_input_dict(batch: List, mean_image: Image):
    return {"images": [mean_image], "texts": list(batch[0])}

def convert_mc_batch_to_vilt_input_dict(batch: List, mean_image: Image):
    texts_a, texts_b = batch[0], batch[1]
    bs = len(texts_a)
    texts_b = list(itertools.chain(*texts_b))
    text_pairs = [[texts_a[i % bs], tb] for i, tb in enumerate(texts_b)]
    return {"images": [mean_image], "texts": text_pairs}
