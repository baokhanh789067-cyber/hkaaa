import torch
import torch.nn as nn
import math

class Adapter(nn.Module):
    """
    【论文升级版】Dual-LoRA Adapter for FedDAT (Fix: Auto-Gating)
    Structure: Input -> Dropout -> Linear(A) -> Linear(B) -> Scaling -> Output
    """

    def __init__(self,
                 names,
                 device,
                 model_dim=768,
                 adapter_reduction_factor=16,
                 lora_alpha=16,       
                 dropout_rate=0.1):   
        super().__init__()
        
        self.r = max(1, model_dim // adapter_reduction_factor)
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.gating = False
        self.input_dim = model_dim
        
        print(f"[Init Dual-LoRA] Rank r={self.r}, Alpha={self.lora_alpha}, Scaling={self.scaling:.2f}, Device={device}")

        if isinstance(names, str):
            names = [names]

        # --- Layer Construction ---
        for name in names:
            if 'adapter' in name:
                # === LoRA A ===
                lora_a = nn.Linear(model_dim, self.r, bias=False).to(device)
                nn.init.kaiming_uniform_(lora_a.weight, a=math.sqrt(5))
                
                # === LoRA B ===
                lora_b = nn.Linear(self.r, model_dim, bias=False).to(device)
                nn.init.zeros_(lora_b.weight)

                setattr(self, f'{name}_down', lora_a) 
                setattr(self, f'{name}_up', lora_b)  

            elif name == 'gating':
                # === Gating ===
                gating_module = nn.Linear(model_dim, 2).to(device)
                nn.init.normal_(gating_module.weight, std=0.02)
                nn.init.zeros_(gating_module.bias)
                setattr(self, f'{name}_module', gating_module)
        
        # [Fix] 强制补全 gating_module，防止 forward 报错
        if not hasattr(self, 'gating_module'):
            gating_module = nn.Linear(model_dim, 2).to(device)
            nn.init.normal_(gating_module.weight, std=0.02)
            nn.init.zeros_(gating_module.bias)
            self.gating_module = gating_module

        # --- Default Freezing ---
        if hasattr(self, 'adapter_1_down'):
            for m in [self.adapter_1_down, self.adapter_1_up]:
                for p in m.parameters():
                    p.requires_grad = False

    def deactivate_gating(self):
        self.gating = False

    def activate_gating(self):
        self.gating = True
        if hasattr(self, 'gating_module'):
            for p in self.gating_module.parameters():
                p.requires_grad = True

    def set_active_adapter(self, name):
        if isinstance(name, str):
            self.active_adapter_down = getattr(self, f'{name}_down')
            self.active_adapter_up = getattr(self, f'{name}_up')

        if name == 'adapter_0':
            for m in [self.adapter_0_down, self.adapter_0_up]:
                for p in m.parameters(): p.requires_grad = True
            if hasattr(self, 'adapter_1_down'):
                for m in [self.adapter_1_down, self.adapter_1_up]:
                    for p in m.parameters(): p.requires_grad = False

        elif name == 'adapter_1':
            if hasattr(self, 'adapter_1_down'):
                for m in [self.adapter_1_down, self.adapter_1_up]:
                    for p in m.parameters(): p.requires_grad = True
            for m in [self.adapter_0_down, self.adapter_0_up]:
                for p in m.parameters(): p.requires_grad = False
        return

    def get_agg_out(self, outs, weights):
        agg_out = weights[:, :, 0] * outs[0]
        for i, out in enumerate(outs[1:]):
            agg_out += weights[:, :, i+1] * out
        return agg_out

    def forward(self, hidden_states, input_tensor=None):
        if input_tensor is None:
            input_tensor = hidden_states

        # Case A: Single Adapter
        if not self.gating:
            x = self.dropout(hidden_states)
            down = self.active_adapter_down(x)
            up = self.active_adapter_up(down) 
            hidden_states = input_tensor + up * self.scaling

        # Case B: Dual-Stream Gating
        else:
            up_outs = []
            indices = []
            if hasattr(self, 'adapter_0_down'): indices.append(0)
            if hasattr(self, 'adapter_1_down'): indices.append(1)

            for i in indices:
                adapter_down = getattr(self, f'adapter_{i}_down')
                adapter_up = getattr(self, f'adapter_{i}_up')
                
                x = self.dropout(hidden_states)
                down_out = adapter_down(x)
                up_out = adapter_up(down_out)
                up_outs.append(up_out * self.scaling)

            # Gating Logic (Safe to call now)
            gate_logits = self.gating_module(input_tensor)
            weight_up = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)

            agg_up_out = self.get_agg_out(up_outs, weight_up)
            hidden_states = input_tensor + agg_up_out

        return hidden_states

# ==========================================
# 🚑 Emergency Fix: Restore missing helper function
# ==========================================
def init_bert_weights(module):
    """
    Initialize the weights for BERT-like models.
    Restored to fix ImportError in adaptered_output.py
    """
    import torch.nn as nn
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
