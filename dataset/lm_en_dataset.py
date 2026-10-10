import json
import random
import re
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import torch

# from sklearn.model_selection import train_test_split
import os
import ast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 构建输入文本
        text = (
            f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        )
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = input_ids != self.tokenizer.pad_token_id

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)

        # 特殊 token
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.inst_start = "[INST]"
        self.inst_end = "[/INST]"

    def __len__(self):
        return len(self.samples)

    def load_data(self, path: str) -> List[Dict]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line.strip()))
        return samples

    # ---------- 构造 prompt ----------
    def build_prompt(self, conversations: List[Dict]) -> str:
        """
        conversations: [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
        """
        prompt_parts = []
        current_turn = ""

        for i, turn in enumerate(conversations):
            role = turn["role"]
            content = turn["content"].strip()

            if role == "user":
                if current_turn:
                    prompt_parts.append(current_turn + " </s>")  # 结束上一轮
                current_turn = f"<s>{self.inst_start} {content} {self.inst_end}"
            elif role == "assistant":
                current_turn += f" {content}"

        if current_turn:
            prompt_parts.append(current_turn + " </s>")

        return "".join(prompt_parts).strip()

    # ---------- 截断策略 ----------
    def truncate_ids(self, token_ids: List[int], target_len: int) -> List[int]:
        """从左右各截一点，保证尾部 answer 一定存在"""
        if len(token_ids) <= target_len:
            return token_ids
        half = target_len // 2
        return token_ids[:half] + token_ids[-(target_len - half) :]

    # ---------- 构造 loss_mask ----------
    def make_loss_mask(self, prompt_ids: List[int]) -> List[int]:
        mask = [0] * len(prompt_ids)

        # 找到 [/INST] 对应的 token 序列
        inst_end_ids = self.tokenizer.encode(self.inst_end, add_special_tokens=False)
        end_len = len(inst_end_ids)

        i = 0
        while i < len(prompt_ids) - end_len + 1:
            if prompt_ids[i : i + end_len] == inst_end_ids:
                # 从 [/INST] 后的第一个 token 开始算 loss
                for j in range(i + end_len, len(prompt_ids)):
                    mask[j] = 1
                break
            i += 1
        # padding 不参与 loss（否则数千个「预测 pad」会把 CE 压到接近 0，loss 虚低）
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            for j, tid in enumerate(prompt_ids):
                if tid == pad_id:
                    mask[j] = 0
        return mask

    # ---------- 取一条样本 ----------
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        prompt_text = self.build_prompt(sample["conversations"])

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_ids = self.truncate_ids(prompt_ids, self.max_length)

        # 补齐到 max_length
        if len(prompt_ids) < self.max_length:
            prompt_ids += [self.tokenizer.pad_token_id] * (
                self.max_length - len(prompt_ids)
            )

        X = torch.tensor(prompt_ids[:-1], dtype=torch.long)
        Y = torch.tensor(prompt_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(self.make_loss_mask(prompt_ids)[1:], dtype=torch.long)

        return X, Y, loss_mask


def _merge_system_user_for_inst(system: Optional[str], user_body: Optional[str]) -> str:
    """
    与 inference.merge_system_and_user_body 一致：将 system 与 user 拼成 [INST] 内单段正文，
    对齐现有英文 SFT 的 `<s>[INST] … [/INST]` 训练格式。
    """
    b = (user_body or "").strip()
    s = (system or "").strip() if system else ""
    if s:
        return f"{s}\n\n{b}"
    return b


class SFTIntentMessagesDataset(SFTDataset):
    """
    读取 MedQA 意图微调 JSON 数组（如 RAG_MedQA_US_train_10178_intent.json）：
    顶层含 `messages`: [system, user, assistant]，将 system+user 合并为 user 槽，
    再按父类逻辑构造与 `SFTDataset` 相同的 Llama-INST 字符串与 loss_mask。
    """

    def __init__(self, json_path: str, tokenizer, max_length: int = 4096):
        Dataset.__init__(self)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.inst_start = "[INST]"
        self.inst_end = "[/INST]"
        self.samples = self._load_messages_json_array(json_path)

    def _load_messages_json_array(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"期望 JSON 顶层为数组: {path}")
        return data

    @staticmethod
    def _parse_messages(messages: List[Dict]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        system_txt, user_txt, assistant_txt = None, None, None
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role == "system":
                system_txt = content
            elif role == "user":
                user_txt = content
            elif role == "assistant":
                assistant_txt = content
        return system_txt, user_txt, assistant_txt

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        msgs = sample.get("messages")
        if not isinstance(msgs, list):
            raise ValueError(f"样本 {idx} 缺少合法 messages 列表")
        system_txt, user_txt, assistant_txt = self._parse_messages(msgs)
        if not user_txt or not assistant_txt:
            raise ValueError(
                f"样本 {idx} 需同时包含 role=user 与 role=assistant 的 content"
            )
        merged_user = _merge_system_user_for_inst(system_txt, user_txt)
        conversations = [
            {"role": "user", "content": merged_user},
            {"role": "assistant", "content": assistant_txt},
        ]
        prompt_text = self.build_prompt(conversations)

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_ids = self.truncate_ids(prompt_ids, self.max_length)

        if len(prompt_ids) < self.max_length:
            prompt_ids += [self.tokenizer.pad_token_id] * (
                self.max_length - len(prompt_ids)
            )

        X = torch.tensor(prompt_ids[:-1], dtype=torch.long)
        Y = torch.tensor(prompt_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(self.make_loss_mask(prompt_ids)[1:], dtype=torch.long)

        return X, Y, loss_mask


class SFTDataset0(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer("<s>assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer("</s>\n", add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn["content"]})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(
                    start + 1, min(end + len(self.eos_id) + 1, self.max_length)
                ):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt = self._create_chat_prompt(sample["conversations"])
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
        self.bos_id = tokenizer("<s>assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer("</s>\n", add_special_tokens=False).input_ids
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item["chosen"]  # 是一个 list，里面包含若干 {role, content}
        rejected = item["rejected"]  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        chosen_input_ids = chosen_encoding["input_ids"]
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding["input_ids"]
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "mask_chosen": mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "mask_rejected": mask_rejected,
        }

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(
                    start + 1, min(end + len(self.eos_id) + 1, self.max_length)
                ):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


if __name__ == "__main__":
    pass
