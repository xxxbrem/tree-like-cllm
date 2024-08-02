import json
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from tqdm import tqdm
import random
import argparse
import transformers
import json
from typing import Optional, Dict, Sequence
import os, sys
import json
import argparse
import numpy as np

def get_default_question(cllm_type):
    if cllm_type == 'sharegpt':
        return "Which methods did Socrates employ to challenge the prevailing thoughts of his time?"
    elif cllm_type == 'spider':
        return "The SQL database has table named vehicle with columns ['Vehicle_ID', 'Model', 'Build_Year', 'Top_Speed', 'Power', 'Builder', 'Total_Production'], table named driver with columns ['Driver_ID', 'Name', 'Citizenship', 'Racing_Series'], table named vehicle_driver with columns ['Driver_ID', 'Vehicle_ID'], Question: What are the vehicle ids and models which have been driven by more than 2 drivers or been driven by the driver named 'Jeff Gordon'?"
    elif cllm_type == 'python':
        return "Implement the Conway's Game of Life. You should start with a 2D grid initialized with some configuration of live and dead cells. 1 for live cell and -1 for dead cell. The simulation should update the grid state by applying the rules for each cell simultaneously: any live cell with fewer than two live neighbors dies, as if by underpopulation. Any live cell with two or three live neighbors lives on to the next generation. Any live cell with more than three live neighbors dies, as if by overpopulation. Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction. initial_grid = [[0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]"
    elif cllm_type == 'gsm8k':
        return "Poppy is solving a 1000-piece jigsaw puzzle. She places a quarter of the pieces on the board, then her mom places a third of the remaining pieces. How many jigsaw pieces are left to be placed?"
    else:
        return "Tell me a short story."

def get_system_prompt(cllm_type):
    if cllm_type == 'sharegpt':
        return "Answer in English unless other language is used. A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    elif cllm_type == 'spider':
        return "Could you translate the following question into SQL. Please only generate SQL, don't include explanation in the answer.\n"
    elif cllm_type == 'python':
        return "Please generate code based on the following doc:\n"
    elif cllm_type == 'gsm8k':
        return ""
    else:
        return "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"

def get_instruction_template(system_prompt, roles, model_input, cllm_type):
    if cllm_type == 'sharegpt':
        return system_prompt + f"{roles[0]}: " + f"{model_input}\n{roles[1]}: "
    if cllm_type == 'spider' or 'python':
        return f"### Instruction:\n" + system_prompt + f"{model_input}\n" + f"### Response:\n"
    if cllm_type == 'gsm8k':
        prompt_mapping = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"
        return prompt_mapping.format(input=model_input)
    else:
        return system_prompt + f"{roles[0]}: " + f"{model_input}\n{roles[1]}: "
    

def detect_repetitive_patterns(tokenizer, prompt_ids, repeat_ngram_size):

    if len(prompt_ids.shape)==1:
        prompt_ids = prompt_ids
    elif len(prompt_ids.shape)==2:
        prompt_ids = prompt_ids[0]
    elif len(prompt_ids.shape)==3:
        prompt_ids = prompt_ids[0][0]
    else:
        print(f'Unexpected shape {prompt_ids.shape}! Please check prompt ids')
        assert False

    count = 1
    for i in range(1, len(prompt_ids)):
        if prompt_ids[i] == tokenizer.eos_token_id:
            break
        if prompt_ids[i] == prompt_ids[i - 1]:
            count += 1
            if count == repeat_ngram_size:
                return True
        else:
            count = 1

    return False

def jacobian_generated_data_postprocessed(generated_data, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    low_quality_data_id_lst = []
    # delete low quality data with repetitive pattern
    for i, d in enumerate(generated_data):
        if detect_repetitive_patterns(tokenizer, np.array(d['prompt_ids']), repeat_ngram_size=10):
            prompt_ids = np.array(d['prompt_ids'])
            if len(prompt_ids.shape)==2:
                prompt_ids = prompt_ids[0]
            elif len(prompt_ids.shape)==3:
                prompt_ids = prompt_ids[0][0]
            print(f'Low quality generation detected: {tokenizer.decode(prompt_ids)}')
            low_quality_data_id_lst.append(i)
    print(f'{len(low_quality_data_id_lst)} low quality data detected. {len(low_quality_data_id_lst)/len(generated_data)} percent of low quality data.')

    # add complete teacher outputs
    teacher_output_inspector = {}
    for d in generated_data:
        data_id = d["data_id"]
        if data_id in teacher_output_inspector.keys():
            all_teacher_output_map = teacher_output_inspector[data_id]
        else:
            all_teacher_output_map = {}
            #print(data_id)
        itr = d["jacobian_itr_id"]
        # handle bsz=1 case only
        all_teacher_output_map[itr] = d["teacher_output_ids"][0]
        teacher_output_inspector[data_id] = all_teacher_output_map

    teacher_output_collector = {}
    for d_id in teacher_output_inspector.keys():
        all_teacher_output_map = teacher_output_inspector[d_id]
        all_itr = [int(s.split('_')[1]) for s in all_teacher_output_map.keys()]
        print(all_itr)
        max_itr = max(all_itr)
        max_itr_s = "itr_" + str(max_itr)
        complete_teacher_output = all_teacher_output_map[max_itr_s]
        teacher_output_collector[d_id] = complete_teacher_output

    f_result = []
    for d in generated_data:
        data_id = d["data_id"]
        complete_teacher_output = teacher_output_collector[data_id]
        d["complete_teacher_output_ids"] = complete_teacher_output
        f_result.append(d)
    
    cleaned_f_result = []
    for i, d in enumerate(generated_data):
        if i in low_quality_data_id_lst:
            continue
        cleaned_f_result.append(d)


    return cleaned_f_result

def get_logits(self, hidden_states):
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [torch.nn.functional.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    return logits.float()

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
):
    """
    Create a causal mask for bi-directional self-attention.

    Args:
        input_ids_shape (torch.Size): The shape of input_ids tensor, typically (batch_size, tgt_len).
        dtype (torch.dtype): The data type of the mask.
        device (torch.device): The device on which the mask will be placed.
        past_key_values_length (int, optional): The length of past key values. Default is 0.

    Returns:
        torch.Tensor: The causal mask tensor.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )

# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expand attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.

    Args:
        mask (torch.Tensor): The attention mask tensor of shape `[bsz, seq_len]`.
        dtype (torch.dtype): The data type of the mask.
        tgt_len (Optional[int], optional): The target sequence length. If None, it defaults to the source sequence length.

    Returns:
        torch.Tensor: The expanded mask tensor.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )

def _prepare_decoder_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_key_values_length, tree_mask
):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            # inputs_embeds.dtype,
            torch.float32,  # [MODIFIED] force to cast to float32
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        ).to(inputs_embeds.device)
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )


    if tree_mask is not None:
        tree_len = tree_mask.size(-1)
        combined_attention_mask[:, :, -tree_len:, -tree_len:][
            tree_mask == 0
            ] = combined_attention_mask.min()

    return combined_attention_mask