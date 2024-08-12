from dataclasses import dataclass, field
import json
import math
import pathlib
import functools
from typing import Dict, Optional, Sequence, List, Tuple
import random
from tqdm import tqdm
import torch.nn.functional as F
import sqlite3
import time
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother, get_module_class_from_name
from fastchat.model.model_adapter import get_conversation_template
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

import torch.nn.functional as F
from transformers import LlamaModel,LlamaForCausalLM
import argparse
from cllm.utils import get_logits, _prepare_decoder_attention_mask

def delete_false_key_value(
        self,
        num_of_false_tokens,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
   
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-num_of_false_tokens, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num_of_false_tokens, :]

@torch.inference_mode()
def jacobi_forward(
    self,
    input_ids: torch.LongTensor = None,
    tokenizer=None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    max_new_tokens: Optional[int] = None,
    prefill_phase: Optional[bool] = False,
    chat: Optional[bool] = False,
):
    assert use_cache == True

    if input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    
    if prefill_phase: # prefill phase, just compute the keys & values of prompt
        # self.model is the instance of class LlamaModel
        inputs_embeds = self.model.embed_tokens(input_ids)
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length) 

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if self.model._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.model._use_sdpa :
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        for decoder_layer in self.model.layers:

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[1]

        hidden_states = self.model.norm(hidden_states)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        predict_next_tokens = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.001,  dim=-1)
        first_correct_token = predict_next_tokens[:, -1]
        return next_decoder_cache, first_correct_token
    else: # generation phase, input as random_initilized point and output as fixed point
        jacobian_trajectory = []
        accurate_n_gram = torch.zeros_like(input_ids).to(input_ids.device)
        accurate_length = 0

        next_point = input_ids
        jacobian_trajectory.append(next_point)

        iter_counter = 0

        prev_len = 0
        while True:

            current_point = next_point
            inputs_embeds = self.model.embed_tokens(current_point)
            attention_mask = None
            position_ids = None
            seq_length = current_point.shape[1]
            if use_cache:
                use_legacy_cache = not isinstance(past_key_values, Cache)
                if use_legacy_cache:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_key_values_length = past_key_values.get_usable_length(seq_length) 
            # print(past_key_values_length) # return previous_seq_length
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)

            if self.model._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            elif self.model._use_sdpa :
                # output_attentions=True can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            else:
                # 4d mask is passed through the layers
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )
            # embed positions
            hidden_states = inputs_embeds

            # decoder layers            
            for decoder_layer in self.model.layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                )

                hidden_states = layer_outputs[0]

            hidden_states = self.model.norm(hidden_states)

            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)

            logits = logits.float()
            all_shift_one_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.001, dim=-1)

            next_point = torch.cat((current_point[0, 0].view(1,-1), all_shift_one_token[0, :seq_length-1].view(1,-1)), dim=-1)

            first_false_index = torch.where(torch.eq(current_point[0], next_point[0]) == False)[0]
            
            jacobian_trajectory.append(next_point)

            if len(first_false_index) > 0:
                fast_forward_cnt = first_false_index[0].item()

                past_key_values.delete_false_key_value(seq_length - fast_forward_cnt) # delete the false keys & values
            else:
                fast_forward_cnt = torch.sum(torch.eq(current_point, next_point)).item()

                accurate_n_gram[0, accurate_length : accurate_length + fast_forward_cnt] = next_point[0, :fast_forward_cnt]         
                first_correct_token = all_shift_one_token[:,-1]   
                if chat:
                    if tokenizer.eos_token_id in accurate_n_gram[0, :accurate_length + fast_forward_cnt]:
                        eos_positions = torch.where(accurate_n_gram[0]==tokenizer.eos_token_id)[0]
                        eos_position = eos_positions[0]
                        generated_str = tokenizer.decode(accurate_n_gram[0, :eos_position], skip_special_tokens=True)
                    else:
                        generated_str = tokenizer.decode(accurate_n_gram[0, :accurate_length + fast_forward_cnt], skip_special_tokens=True)

                    print(generated_str[prev_len:], flush=True, end="")
                    prev_len = len(generated_str)
                break 

            accurate_n_gram[0, accurate_length : accurate_length + fast_forward_cnt] = next_point[0, :fast_forward_cnt]
            accurate_length += fast_forward_cnt
            next_point = next_point[0, fast_forward_cnt:].view(1,-1) # only false tokens should be re-generated

            if chat:
                if tokenizer.eos_token_id in accurate_n_gram[0, :accurate_length]:
                    eos_positions = torch.where(accurate_n_gram[0]==tokenizer.eos_token_id)[0]
                    eos_position = eos_positions[0]

                    generated_str = tokenizer.decode(accurate_n_gram[0, :eos_position], skip_special_tokens=True)
                else:
                    generated_str = tokenizer.decode(accurate_n_gram[0, :accurate_length], skip_special_tokens=True)

                print(generated_str[prev_len:], flush=True, end="")
                prev_len = len(generated_str)

            iter_counter += 1

        return accurate_n_gram, first_correct_token, iter_counter, accurate_length


@torch.inference_mode()
def jacobi_forward_profiling(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    max_new_tokens: Optional[int] = None,
    prefill_phase: Optional[bool] = False,
):
    
    assert use_cache == True

    if input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    
    if prefill_phase: # prefill phase, just compute the keys & values of prompt
        # self.model is the instance of class LlamaModel
        inputs_embeds = self.model.embed_tokens(input_ids)
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length) 

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if self.model._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.model._use_sdpa :
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        for decoder_layer in self.model.layers:

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[1]

        hidden_states = self.model.norm(hidden_states)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        predict_next_tokens = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)
        first_correct_token = predict_next_tokens[:, -1]
        return next_decoder_cache, first_correct_token
    else: # generation phase, input as random_initilized point and output as fixed point
        jacobian_trajectory = []
        accurate_n_gram = torch.zeros_like(input_ids).to(input_ids.device)
        accurate_length = 0
        next_point = input_ids
        jacobian_trajectory.append(next_point)

        iter_counter = 0
        while True:

            current_point = next_point
            inputs_embeds = self.model.embed_tokens(current_point)
            attention_mask = None
            position_ids = None
            seq_length = current_point.shape[1]
            if use_cache:
                use_legacy_cache = not isinstance(past_key_values, Cache)
                if use_legacy_cache:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_key_values_length = past_key_values.get_usable_length(seq_length) 
            # print(past_key_values_length) # return previous_seq_length
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)

            if self.model._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            elif self.model._use_sdpa :
                # output_attentions=True can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            else:
                # 4d mask is passed through the layers
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )
            # embed positions
            hidden_states = inputs_embeds

            # decoder layers            
            for decoder_layer in self.model.layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                )

                hidden_states = layer_outputs[0]

            hidden_states = self.model.norm(hidden_states)

            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)

            logits = logits.float()
            all_shift_one_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.01, dim=-1)
            next_point= torch.cat((current_point[0, 0].view(1,-1), all_shift_one_token[0, :seq_length-1].view(1,-1)), dim=-1)
            jacobian_trajectory.append(next_point)
            
            if torch.all(torch.eq(current_point, next_point)).item():    
                #print('Successfully break!')
                #print(next_point)
                first_correct_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)[:,-1]
                break
            past_key_values.delete_false_key_value(seq_length)

            iter_counter += 1

        return jacobian_trajectory[:-1], next_point, first_correct_token, iter_counter
    
@torch.inference_mode()
def topK_genrate(self, input_ids, max_new_tokens, past_key_values, use_cache=True, top_k=3, total_tokens=63, depth=2):
    if input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    jacobian_trajectory = []
    next_point = input_ids
    jacobian_trajectory.append(next_point)

    iter_counter = 0
    correct_token_index = 1
    tree_flag = False
    while True:

        current_point = next_point
        inputs_embeds = self.model.embed_tokens(current_point) if not tree_flag else self.model.embed_tokens(tree_candidates) 
        attention_mask = None
        position_ids = None if not tree_flag else position_ids
        seq_length = inputs_embeds.shape[1]
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length) 
        # print(past_key_values_length) # return previous_seq_length
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = tree_position_ids + past_key_values_length
        
        if tree_flag:
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length + past_key_values_length), dtype=torch.bool, device=inputs_embeds.device
                )
                attention_mask = _prepare_decoder_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length, tree_mask
                )
                # attention_mask = tree_mask
        elif self.model._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.model._use_sdpa :
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers            
        for decoder_layer in self.model.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]
        # for idx, decoder_layer in enumerate(self.model.layers):
        #     past_key_value = (
        #         past_key_values[idx] if past_key_values is not None else None
        #     )

        #     layer_outputs = decoder_layer(
        #         hidden_states,
        #         attention_mask=attention_mask,
        #         position_ids=position_ids,
        #         past_key_value=past_key_value,
        #         use_cache=use_cache,
        #     )

        #     hidden_states = layer_outputs[0]

        hidden_states = self.model.norm(hidden_states)


        logits = get_logits(self, hidden_states)
        
        if tree_flag:
            logits = logits[0, retrieve_indices]
            print(f"logits:\n{logits}")
        # if iter_counter == 0:
            # topk_token = torch.topk(torch.nn.functional.softmax(logits, dim=-1)/0.01, top_k, dim=-1).indices[:, 0, :]
            # torch.Size([1, 3])
        # all_shift_one_token = generate(logits, current_point, past_key_values, top_k=3, total_tokens=63, depth=3)
        
        topk_token = torch.topk(torch.nn.functional.softmax(logits, dim=-1)/0.01, top_k, dim=-1).indices
        # torch.nonzero(topk_token[0] == 29889, as_tuple=False)
        all_shift_one_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.01, dim=-1)
        next_point = torch.cat((current_point[0, 0].view(1,-1), all_shift_one_token[0, :seq_length-1].view(1,-1)), dim=-1)
        jacobian_trajectory.append(next_point)
        
        # for repeatitive tokens
        # current_token = -1
        # for i in range(correct_token_index, seq_length-1):
        #     if current_token == next_point[:, i]:
        #         possible_point = torch.zeros((1, seq_length+4), device=next_point.device, dtype=int)
        #         possible_point[0][:i-1] = next_point[0][:i-1]
        #         possible_point[0][i-1:i+2] = topk_token[0][i-1][:3]
        #         possible_point[0][i+2:i+5] = topk_token[0][i][:3]
        #         possible_point[0][i+5:] = next_point[0][i+1:]
        #         tree_flag = True
        #         position_ids = torch.zeros_like(possible_point, device=position_ids.device)
        #         position_ids[0][:i-1] = torch.arange(0, i-1)
        #         position_ids[0][i-1:i+2] = torch.tensor([i-1]).repeat(3)
        #         position_ids[0][i+2:i+5] = torch.tensor([i]).repeat(3)
        #         position_ids[0][i+5:] = torch.arange(i+5, seq_length+4)
                
        #         retrieve_indices = torch.cat((position_ids[:, :i], position_ids[:, i+4:]), dim=1).repeat(9, 1)
        #         retrieve_indices[:, i-1] = torch.tensor([i-1, i, i+1], device=retrieve_indices.device).repeat_interleave(3)
        #         retrieve_indices[:, i] = torch.arange(i+2, i+5, device=retrieve_indices.device).repeat(3)
        #         break
        #     current_token = next_point[:, i]
        
        # print(f"itr: {iter_counter}")
        # print(next_point)
        # torch.topk(torch.nn.functional.softmax(logits, dim=-1)/0.01, 3, dim=-1)
        
        # for short fastforward
        

        if not tree_flag:
        
            if torch.all(torch.eq(current_point, next_point)).item():    
                first_correct_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)[:,-1]
                break
            past_key_values.delete_false_key_value(seq_length)
            
            correct_token_index = int(torch.where((current_point == next_point) == False)[1][0])
        iter_counter += 1
        
        if correct_token_index > 1:
            # correct_token_index
            candidates_num = correct_token_index + 1 + (top_k**(depth+1)-top_k)//(top_k-1) + (max_new_tokens-correct_token_index-1-depth)*9
            tree_candidates = torch.zeros((batch_size, candidates_num), device=next_point.device, dtype=int)
            tree_position_ids = torch.zeros((batch_size, candidates_num), device=next_point.device, dtype=int)
            retrieve_indices = torch.zeros((batch_size, top_k**depth, max_new_tokens), device=next_point.device, dtype=int)
            tree_mask = torch.zeros((batch_size, batch_size, candidates_num, candidates_num), device=next_point.device, dtype=int)
            # batch_size = 1
            tree_candidates[batch_size-1][:correct_token_index+1] = next_point[batch_size-1][:correct_token_index+1]
            tree_position_ids[batch_size-1][:correct_token_index+1] = torch.arange(0, correct_token_index+1, device=next_point.device)
            retrieve_indices[batch_size-1][:, :correct_token_index+1] = tree_position_ids[batch_size-1][:correct_token_index+1].repeat(top_k**depth, 1)
            tree_mask[batch_size-1][batch_size-1][:,:] = torch.tril(torch.ones(candidates_num, candidates_num))
            
            start = correct_token_index+1
            end = correct_token_index+1+top_k
            
            for j in range(depth):
                
                tree_candidates[batch_size-1][start:end] = topk_token[batch_size-1][correct_token_index+j][:top_k].repeat_interleave(top_k*j if j > 0 else 1)
                tree_position_ids[batch_size-1][start:end] = torch.tensor([correct_token_index+1+j] * (end-start), device=next_point.device)
                retrieve_indices[batch_size-1][:, correct_token_index+1+j] = torch.arange(start, end, device=next_point.device).repeat_interleave(top_k**(depth-j-1))
                tree_mask[batch_size-1][batch_size-1][start:, start:start+top_k**(j+1)] = torch.eye(top_k**(j+1), top_k**(j+1)).repeat((candidates_num-start)//top_k**(j+1), 1)

                start = end
                end += top_k**(j+2)
            
            tree_flag = True
            remain_len = max_new_tokens-(correct_token_index+1+depth)
            tree_candidates[batch_size-1][start:] = next_point[batch_size-1][correct_token_index+1+depth:].repeat_interleave(top_k**depth)
            tree_position_ids[batch_size-1][start:] = torch.arange(tree_position_ids[0][start-1]+1, tree_position_ids[0][start-1]+1+remain_len).repeat_interleave(top_k**depth)
            retrieve_indices[batch_size-1][:, correct_token_index+1+depth:] = torch.arange(start, candidates_num).reshape(remain_len, -1).T
            for _ in range(remain_len):
                tree_mask[batch_size-1][batch_size-1][start:, start:start+top_k**(depth)] = torch.eye(top_k**(depth), top_k**(depth)).repeat((candidates_num-start)//top_k**(depth), 1)
                start += top_k**depth

    return jacobian_trajectory[:-1], next_point, first_correct_token, iter_counter