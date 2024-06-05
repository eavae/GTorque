import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict
from tqdm import tqdm
from collections import defaultdict


class PPL:

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_size: int = 4,
        max_length: int = 512,
        stride: int = 256,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._stride = stride
        self._batch_size = batch_size

    def __call__(
        self,
        file_paths: List[str],
        show_progress=True,
    ) -> Dict[str, float]:
        # tokenize
        tokens: List[Tuple[str, torch.Tensor]] = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                encodings = self._tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self._max_length,
                )

                seq_len = encodings.input_ids.size(1)
                prev_end_loc = 0
                for begin_loc in range(0, seq_len, self._stride):
                    end_loc = min(begin_loc + self._max_length, seq_len)
                    if end_loc - begin_loc < self._max_length:
                        begin_loc = max(0, end_loc - self._max_length)
                    mask_len = prev_end_loc - begin_loc

                    input_ids = encodings.input_ids[:, begin_loc:end_loc]
                    target_ids = input_ids.clone()
                    target_ids[:, :-mask_len] = -100

                    prev_end_loc = end_loc
                    tokens.append((file_path, input_ids, target_ids))

                    if end_loc >= seq_len:
                        break

        # batch compute ppl
        self._model.eval()
        loss_fct = torch.nn.CrossEntropyLoss()
        ppls = defaultdict(list)
        for i in tqdm(
            range(0, len(tokens), self._batch_size),
            disable=not show_progress,
            desc="Computing PPL",
        ):
            batch_tokens = tokens[i : i + self._batch_size]  # noqa
            file_paths = [token[0] for token in batch_tokens]
            input_ids = torch.cat([token[1] for token in batch_tokens], dim=0)
            labels = torch.cat([token[2] for token in batch_tokens], dim=0)

            outputs = self._model(
                input_ids=input_ids,
                labels=labels,
            )

            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            for batch_idx, file_path in enumerate(file_paths):
                loss = loss_fct(
                    shift_logits[batch_idx].view(-1, shift_logits.size(-1)),
                    shift_labels[batch_idx].view(-1),
                )
                ppl = torch.exp(loss)
                ppls[file_path].append(ppl.item())

        return {file_path: sum(ppl) / len(ppl) for file_path, ppl in ppls.items()}
