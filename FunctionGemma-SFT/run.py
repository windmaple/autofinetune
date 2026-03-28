import os
import json
import logging
import re
import shutil
import functools
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from huggingface_hub import snapshot_download, hf_hub_download
from datasets import load_dataset
from transformers import AutoTokenizer

# Tunix imports
from tunix.models.gemma3 import params as gemma_params
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.generate import sampler as sampler_lib
from tunix.sft import peft_trainer
from tunix.sft import metrics_logger
from tunix.sft import utils
from tunix.generate import tokenizer_adapter as tokenizer_lib
import qwix
import random

"""## Setup

Set up some configs and constants.
"""

# Constants
MODEL_ID = "google/functiongemma-270m-it"
DATASET_ID = "google/mobile-actions"
OUTPUT_DIR = os.path.abspath("./mobile-actions-tunix")
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
BATCH_SIZE = 8
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
MAX_LENGTH = 1024
EVAL_EVERY_N_STEPS = 50
LORA_RANK = 8
LORA_ALPHA = 16
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# save RAM
#os.environ['XLA_FLAGS'] = "--xla_llvm_disable_expensive_passes=true"

# XLA cache
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

"""## Helper function to extract function calls"""

def extract_function_call(model_output):
    results = []
    call_pattern = r"<start_function_call>(.*?)<end_function_call>"
    raw_calls = re.findall(call_pattern, model_output, re.DOTALL)

    for raw_call in raw_calls:
        if not raw_call.strip().startswith("call:"):
            continue
        try:
            pre_brace, args_segment = raw_call.split("{", 1)
            function_name = pre_brace.replace("call:", "").strip()
            args_content = args_segment.strip()
            if args_content.endswith("}"):
                args_content = args_content[:-1]
            arguments = {}
            arg_pattern = r"(?P<key>[^:,]*?):<escape>(?P<value>.*?)<escape>"
            arg_matches = re.finditer(arg_pattern, args_content, re.DOTALL)
            for match in arg_matches:
                key = match.group("key").strip()
                value = match.group("value")
                arguments[key] = value
            if function_name:
                results.append({"function": {"name": function_name, "arguments": arguments}})
        except ValueError:
            continue
    return results

"""# Download the dataset"""

print("Downloading model and dataset...")
local_model_path = snapshot_download(repo_id=MODEL_ID, ignore_patterns=["*.pth"])
data_file = hf_hub_download(repo_id=DATASET_ID, filename="dataset.jsonl", repo_type="dataset")
dataset = load_dataset("text", data_files=data_file, encoding="utf-8")["train"].shuffle(seed=SEED)

train_data = dataset.filter(lambda x: json.loads(x['text'])['metadata'] == 'train')
full_eval = dataset.filter(lambda x: json.loads(x['text'])['metadata'] == 'eval')
eval_data_for_acc = full_eval
val_data_for_loss = full_eval

"""## Prepare tokenizer, model and sampler"""

tokenizer = AutoTokenizer.from_pretrained(local_model_path, fix_mistral_regex=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

def get_model_config(config_path):
    config = gemma_lib.ModelConfig.gemma3_270m()
    return config

config_path = os.path.join(local_model_path, "config.json")
model_config = get_model_config(config_path)

NUM_TPUS = len(jax.devices())
MESH = [(1, NUM_TPUS), ("fsdp", "tp")] if NUM_TPUS > 1 else [(1, 1), ("fsdp", "tp")]
mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))

with mesh:
    base_model = params_safetensors_lib.create_model_from_safe_tensors(local_model_path, model_config, mesh)
    lora_provider = qwix.LoraProvider(
        module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
        rank=LORA_RANK, alpha=LORA_ALPHA,
    )
    model_input = base_model.get_model_input()
    model = qwix.apply_lora_to_model(base_model, lora_provider, rngs=nnx.Rngs(SEED), **model_input)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    print("LoRA applied and sharded.")

sampler = sampler_lib.Sampler(
    transformer=model, tokenizer=tokenizer,
    cache_config=sampler_lib.CacheConfig(cache_size=4096, num_layers=model_config.num_layers, num_kv_heads=model_config.num_kv_heads, head_dim=model_config.head_dim)
)
STOP_IDS = [1, 106, 50, tokenizer.eos_token_id]

"""## Run a single sample and full evalation before finetune

We run a single sample to see how FunctionGemma performs. It is doing OK but calling the right tool, but the body parameter is not very good ('<escape>The body of the email.<escape>').

We also run the full evaluation on the eval dataset. FunctionGemma achieves ~65% accurary. Not bad, but we can do better.
"""

def run_eval(data_subset, label):
    print(f"--- {label} ---")
    correct_count, total_count = 0, 0
    for i, example in enumerate(data_subset):
        orig_data = json.loads(example['text'])
        messages = orig_data['messages']
        prompt = tokenizer.apply_chat_template(messages[:-1], tools=orig_data['tools'], tokenize=False, add_generation_prompt=True)
        try:
            out = sampler([prompt], max_generation_steps=MAX_LENGTH, eos_tokens=STOP_IDS, seed=SEED)
            model_output = out.text[0]
        except Exception as e:
            print(f"Error: {e}")
            continue
        output_fc = extract_function_call(model_output)
        target_fc = messages[2].get('tool_calls', [])
        target_names = [fc['function']['name'] for fc in target_fc]
        output_names = [fc['function']['name'] for fc in output_fc]
        target_args = [dict(sorted(fc['function']['arguments'].items())) for fc in target_fc]
        output_args = [dict(sorted(fc['function']['arguments'].items())) for fc in output_fc]
        if (target_names == output_names) and (target_args == output_args):
            correct_count += 1
        total_count += 1
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(data_subset)} - Accuracy: {correct_count/total_count:.2%}")
    acc = correct_count/total_count if total_count > 0 else 0
    print(f"Final {label} Accuracy: {acc:.2%}")
    return acc

"""## Finetune the model

Tunix has certain expectations on the input data, so we create a `CustomDataset` for Tunix and prepare the dataset accordingly.
"""

class CustomDataset:
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.data)

    def __iter__(self):
        for item in self.data:
            template_inputs = json.loads(item['text'])
            prompt_and_completion = self.tokenizer.apply_chat_template(
                template_inputs['messages'], tools=template_inputs['tools'], tokenize=False, add_generation_prompt=False
            )
            prompt_only = self.tokenizer.apply_chat_template(
                template_inputs['messages'][:-1], tools=template_inputs['tools'], tokenize=False, add_generation_prompt=True
            )

            tokenized_full = self.tokenizer(prompt_and_completion, add_special_tokens=False)
            tokenized_prompt = self.tokenizer(prompt_only, add_special_tokens=False)

            full_ids = tokenized_full['input_ids']
            prompt_len = len(tokenized_prompt['input_ids'])

            if len(full_ids) > self.max_length:
                full_ids = full_ids[:self.max_length]

            input_tokens = np.full((self.max_length,), self.tokenizer.pad_token_id, dtype=np.int32)
            input_tokens[:len(full_ids)] = full_ids

            input_mask = np.zeros((self.max_length,), dtype=np.int32)
            if len(full_ids) > prompt_len:
                mask_end = min(len(full_ids), self.max_length)
                input_mask[prompt_len:mask_end] = 1

            yield peft_trainer.TrainingInput(
                input_tokens=jnp.array(input_tokens, dtype=jnp.int32),
                input_mask=jnp.array(input_mask, dtype=jnp.int32)
            )

def data_generator(split_data, batch_size):
    dataset_obj = CustomDataset(split_data, tokenizer, MAX_LENGTH)
    batch_tokens, batch_masks = [], []
    for item in dataset_obj:
        batch_tokens.append(item.input_tokens)
        batch_masks.append(item.input_mask)
        if len(batch_tokens) == batch_size:
            yield peft_trainer.TrainingInput(input_tokens=jnp.array(np.stack(batch_tokens)), input_mask=jnp.array(np.stack(batch_masks)))
            batch_tokens, batch_masks = [], []

print("Preparing training data...")
train_batches = list(data_generator(train_data, BATCH_SIZE))
val_batches = list(data_generator(val_data_for_loss, BATCH_SIZE))

"""Now we kick off the finetuning. Tunix integrates seamlessly with TensorBoard and Weight and Biases, so that we can visualize the training progress."""

def gen_model_input_fn(x: peft_trainer.TrainingInput):
    pad_mask = x.input_tokens != tokenizer.pad_token_id
    positions = utils.build_positions_from_mask(pad_mask)
    attention_mask = utils.make_causal_attn_mask(pad_mask)
    return {'input_tokens': x.input_tokens, 'input_mask': x.input_mask, 'positions': positions, 'attention_mask': attention_mask}

print("Starting Training...")
max_steps = len(train_batches) * NUM_EPOCHS
lr_schedule = optax.cosine_decay_schedule(init_value=LEARNING_RATE, decay_steps=max_steps)
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir=os.path.join(OUTPUT_DIR, "logs"), flush_every_n_steps=10
)
training_config = peft_trainer.TrainingConfig(
    eval_every_n_steps=EVAL_EVERY_N_STEPS,
    max_steps=max_steps,
    checkpoint_root_directory=os.path.join(OUTPUT_DIR, "ckpts"),
    metrics_logging_options=metrics_logging_options,
)
trainer = peft_trainer.PeftTrainer(model, optax.adamw(lr_schedule), training_config).with_gen_model_input_fn(gen_model_input_fn)

with mesh:
    trainer.train(train_batches, val_batches)
print("Training Complete.")

"""## Post-train evaluation

Now we run the same test sample gain and this time the response is better (body is now 'Don't forget to finalize your quarterly goals before the meeting.').

And the accuracy reaches ~88% after just one epoch of finetune.
"""

run_eval(eval_data_for_acc, "Post-train Eval")

"""## Summary

Congratulation! You have finetuned the FunctionGemma model successfully.
"""
