import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import get_polynomial_decay_schedule_with_warmup
from tqdm import tqdm
from typing import List, Dict
from torch.optim import AdamW
from src.utils.vqa_utils import FeatureHook

class TaskTrainer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.kl_criterion = kl_loss

    def get_hook(self, layer):
        # [Fix] LoRA 模式下禁用 hook，直接返回 None，防止 AttributeError
        return None, None, None, None

    # [Fix] 添加 comm_round 参数以匹配 main.py 的调用
    def train(self, model, comm_round=None, er=None, ewc=None, der=None, derpp=None, pnn=None, hat=None) -> (float, Dict):
        """
        Trains model on VQA task
        """
        # 兼容性处理：如果 comm_round 没传进来，代码里也没用到，但为了签名匹配必须要有
        
        # 为了防止 model.module 报错（单机多卡 vs 单卡），尝试解包
        if hasattr(model, "module"):
            unwrapped_model = model.module
        else:
            unwrapped_model = model

        # 初始化 FedDAT 需要的 adapter 参数 (Teacher <- Student)
        for name in model.state_dict().keys():
            if 'adapter_1' in name: # Global
                p = model.state_dict()[name].data.clone().detach()
                name_tgt = name.replace('adapter_1', 'adapter_2') # Backup/Teacher
                if name_tgt in model.state_dict().keys():
                    model.state_dict()[name_tgt].data.copy_(p)

        # 冻结 adapter_2 (Teacher)
        for n, p in model.named_parameters():
            if 'adapter_2' in n:
                p.requires_grad = False

        if not os.path.isdir(self.task_output_dir):
            os.makedirs(self.task_output_dir, exist_ok=True)

        model = self.accelerator.prepare(model)

        optimizer = self.create_optimizer(model, self.args.optimizer_mode)
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.max_steps * self.warmup_ratio),
            num_training_steps=self.max_steps,
            lr_end=0,
            power=1,
        )

        best_score = 0.
        model.zero_grad()
        optimizer, scheduler = self.accelerator.prepare(optimizer, scheduler)

        if 'nlvr2' in self.task_key:
            loader = self.nlvr_train_dataloader
        elif 'snli-ve' in self.task_key:
            loader = self.snli_ve_train_dataloader
        elif 'vcr' in self.task_key:
            loader = self.vcr_train_dataloader
        else:
            loader = self.vqa_train_dataloader

        for epoch in range(self.local_epochs):
            model.train()
            for step, batch in enumerate(
                tqdm(
                    loader,
                    desc="Training epoch {}".format(epoch + 1),
                )
            ):
                if self.args.debug > 0 and step > self.args.debug:
                    break

                if "vilt" not in self.args.encoder_name:
                    batch = self.add_alpha(epoch, batch, step)

                loss = self.train_step(
                    model, step, batch, optimizer, scheduler,
                    hooks=None, # [Fix] 显式传入 None
                    epoch=epoch
                )

                if (
                    self.args.do_wandb_logging
                    and (step + 1) % self.args.wandb_freq == 0
                ):
                    log_dict = {self.task_key: {"loss": loss.item()}}
                    self.accelerator.log(log_dict)

        eval_score = 0.
        self.accelerator.wait_for_everyone()

        del loss, optimizer, scheduler
        unwrapped_model = self.accelerator.unwrap_model(model)

        torch.cuda.empty_cache()
        self.accelerator.free_memory()

        return eval_score, unwrapped_model

    def eval_one_loader(self, model, loader):
        model.eval()
        model = self.accelerator.prepare(model)
        samples_seen = 0
        if self.args.encoder_name in ["vilt", "viltbert"]:
            eval_score = 0
            for step, batch in enumerate(
                tqdm(loader, desc=f"Evaluating on {self.task_key} val set")
            ):
                if self.args.debug > 0 and step > self.args.debug:
                    break
                output = self.forward_pass(model, batch, do_eval=True)
                logits = output[1]

                if self.task_key in ['nlvr2', 'snli-ve', 'vcr']:
                    target = batch['labels'].to(self.device)
                    logits, target = self.accelerator.gather((logits, target))
                    if step == len(loader) - 1:
                        logits = logits[
                            : len(loader.dataset) - samples_seen
                        ]
                        target = target[
                            : len(loader.dataset) - samples_seen
                        ]
                    else:
                        samples_seen += target.shape[0]
                    batch_scores = (logits.argmax(-1).cpu() == target.cpu())
                    eval_score += batch_scores.sum().item()

                else:
                    target = batch["target_scores"].to(self.device)
                    logits, target = self.accelerator.gather((logits, target))
                    if step == len(loader) - 1:
                        logits = logits[
                            : len(loader.dataset) - samples_seen
                        ]
                        target = target[
                            : len(loader.dataset) - samples_seen
                        ]
                    else:
                        samples_seen += target.shape[0]
                    answer_scores = self.compute_score_with_logits(logits, target)
                    batch_scores = torch.sum(answer_scores, 1)
                    eval_score += batch_scores.sum().item()
            eval_score = eval_score / len(loader.dataset) * 100.0
        else:
            result = []
            eval_score = 0.
            for step, (image, question, gts) in enumerate(
                tqdm(loader, desc="Evaluating on VQA val set")
            ):
                if self.args.debug > 0 and step > self.args.debug:
                    break
                batch = {
                    "images": image,
                    "questions": question,
                    "answer_list": loader.dataset.answer_list,
                    "train": False,
                    "k": 64,
                }

                output = model(self.task_key, batch)
                topk_ids = output[0]
                topk_probs = output[1]

                topk_ids, topk_probs, gts = self.accelerator.gather(
                    (topk_ids, topk_probs, gts )
                )
                if step == len(loader) - 1:
                    topk_ids = topk_ids[
                        : len(loader.dataset) - samples_seen
                    ]
                    topk_probs = topk_probs[
                        : len(loader.dataset) - samples_seen
                    ]
                    gts = gts[
                        : len(loader.dataset) - samples_seen
                    ]
                else:
                    samples_seen += len(gts)
                for topk_id, topk_prob, gt in zip(
                    topk_ids, topk_probs, gts
                ):
                    _, pred = topk_prob.max(dim=0)
                    pred_ans = topk_id[pred].unsqueeze(0)
                    if len(gt)>0:
                        pred_ans = torch.stack([pred_ans]*len(gt))
                    eval_score += int(torch.any(torch.eq(pred_ans, gt)))
            eval_score = eval_score / len(loader.dataset) * 100.0

        model.train()
        torch.cuda.empty_cache()
        self.accelerator.free_memory()
        return eval_score

    def eval(self, model):
        """
        Evaluates model on VQA validation set
        Returns validation VQA score
        """

        if 'nlvr2' in self.task_key:
            loader = self.nlvr_val_dataloader
        elif 'snli-ve' in self.task_key:
            loader = self.snli_ve_dev_dataloader
        elif 'vcr' in self.task_key:
            loader = self.vcr_val_dataloader
        else:
            if 'gqa' in self.task_key:
                loader = self.vqa_val_dataloader
            else:
                loader = self.vqa_test_dataloader

        #set local adapter
        if "dat" in self.args.optimizer_mode:
            # model.module.activate_gating() if ddp else ...
            # safe calling convention
            if hasattr(model, "module"):
                model.module.activate_gating()
            else:
                model.activate_gating()
                
        elif "adapter" in self.args.optimizer_mode:
            if hasattr(model, "module"):
                model.module.set_active_adapter('adapter')
            else:
                model.set_active_adapter('adapter')
                
        eval_score = self.eval_one_loader(model, loader)

        if 'dat' in self.args.optimizer_mode:
            if hasattr(model, "module"):
                m = model.module
            else:
                m = model
                
            m.deactivate_gating()
            m.set_active_adapter('adapter_0')
            eval_score_0 = self.eval_one_loader(model, loader)

            m.deactivate_gating()
            m.set_active_adapter('adapter_1')
            eval_score_1 = self.eval_one_loader(model, loader)
            return [eval_score, eval_score_0, eval_score_1]

        return eval_score

    def forward_pass(self, model, batch, do_eval: bool = False) -> tuple:
        inputs = self.batch2inputs_converter(batch)
        if "albef" in self.args.encoder_name and not do_eval:
            inputs["train"] = True

        if do_eval is True:
            with torch.no_grad():
                if self.args.encoder_name in ["vilt", "viltbert"]:
                    output = model(task_key=self.task_key, **inputs)
                else:
                    output = model(self.task_key, inputs)
        else:
            if self.args.encoder_name in ["vilt", "viltbert"]:
                output = model(task_key=self.task_key, **inputs)
            else:
                output = model(self.task_key, inputs)
        return output

    def train_step(
        self,
        model,
        step,
        batch,
        optimizer=None,
        scheduler=None,
        hooks=None,
        epoch=None,
    ):
        if isinstance(batch, dict) and "target_scores" in batch.keys():
            target = batch["target_scores"].to(self.device)

        # Helper to get underlying model
        if hasattr(model, "module"):
            unwrapped = model.module
        else:
            unwrapped = model

        #train vilt and viltbert
        if "dat" in self.args.optimizer_mode:
            loss_0 = 0.
            # 1. Get logits of all adapters (using Teacher/Gating if enabled, or default)
            with torch.no_grad():
                unwrapped.activate_gating()
                _, logits_all = self.forward_pass(
                    model, batch, do_eval=False
                )

            # 2. Update only server adapter (adapter_1) - Teacher Path
            unwrapped.deactivate_gating()
            unwrapped.set_active_adapter('adapter_1')

            output_1_0, logits_1 = self.forward_pass(
                model, batch, do_eval=False
            )
            
            if "albef" in self.args.encoder_name:
                loss_1 = output_1_0
            else:
                loss_1 = self.loss_criterion(logits_1, target) * target.shape[1]
                
            # Distillation Loss (Global learns from gated/teacher)
            loss_kl_1 = self.kl_criterion(logits_1, logits_all.clone().detach())
            L_1 = (loss_1 + loss_kl_1)/2
            
            self.accelerator.backward(L_1)

            if optimizer is not None:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            # 3. Train local adapter (adapter_0) - Student Path
            unwrapped.activate_gating()
            unwrapped.set_active_adapter('adapter_0')
            
            output_0_0, logits_0 = self.forward_pass(
                model, batch, do_eval=False
            )
            
            if "albef" in self.args.encoder_name:
                loss_0 = output_0_0
            else:
                loss_0 = self.loss_criterion(logits_0, target) * target.shape[1]
            
            # Distillation Loss (Local learns from Global)
            loss_kl_0 = self.kl_criterion(logits_0, logits_1.clone().detach())
            L_0 = (loss_0 + loss_kl_0)/2

            self.accelerator.backward(L_0)
            if optimizer is not None:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

            return loss_0

        else:
            if "adapter" in self.args.optimizer_mode:
                if hasattr(model, "module"):
                    model.module.set_active_adapter('adapter')
                else:
                    model.set_active_adapter('adapter')

            output_0, logits = self.forward_pass(
                model, batch, do_eval=False
            )
            if "albef" in self.args.encoder_name:
                loss = output_0
            else:
                loss = self.loss_criterion(logits, target) * target.shape[1]

            self.accelerator.backward(loss)
            if optimizer is not None:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
            return loss

        return -1

    def create_optimizer(self, model, mode='full'):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (not any(nd in n for nd in no_decay)) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (any(nd in n for nd in no_decay)) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            eps=self.adam_epsilon,
            betas=(0.9, 0.98),
        )
        return optimizer

def kl_loss(output, target, temp=3):
    if output.shape[-1]>3000:
        p = F.log_softmax(output / temp, dim=-1)
        q = F.softmax(target / temp, dim=-1)
    else:
        p = F.log_softmax(output / temp, dim=1)
        q = F.softmax(target / temp, dim=1)

    l_kl = F.kl_div(p, q, reduction="batchmean")
    l_kl = l_kl * temp**2
    return l_kl
