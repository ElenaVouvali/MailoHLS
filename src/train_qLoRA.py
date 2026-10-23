#-----------------------------------------------------------
#                     Imports And Set GPU
#-----------------------------------------------------------

import os
import random
import numpy as np
import torch

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from collections import Counter
from transformers import TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch.nn.functional as F
# import wandb

GPU_ID = 0
torch.cuda.set_device(GPU_ID)
print("Using CUDA device:", GPU_ID, torch.cuda.get_device_name(GPU_ID))


#-----------------------------------------------------------
#            Load Dataset And Kernel-Family-Level Splitting
#-----------------------------------------------------------

JSONL_PATH = "/home/elvouvali/LLM_data/all_kernels_llm_data_filtered.jsonl"

dataset_dict = load_dataset("json", data_files={"raw": JSONL_PATH})
ds_all = dataset_dict["raw"]

# Find easily in the json file the example
ds_all = ds_all.add_column("json_id", list(range(len(ds_all))))

def family_id_from_kernel_name(name: str) -> str:
    s = name.strip().replace("-", "_")  # normalize

    # MachSuite: treat each kernel as its own family (8 families)
    if s.startswith("machsuite_"):
        return name  # keep original label

    # SPCL: one family
    if s.startswith("spcl_example"):
        return "spcl_example"

    # Serrano: one family
    if s.startswith("serrano_") or s.startswith("serrano-"):
        return "serrano"

    # Rodinia: group variants into algorithm families
    if s.startswith("rodinia_"):
        rest = s[len("rodinia_"):]  # e.g. "knn_4_doublebuffer_0"
        if rest.startswith("cfd_flux"):
            return "rodinia_cfd_flux"
        if rest.startswith("cfd_step_factor"):
            return "rodinia_cfd_step_factor"
        if rest.startswith("lc_gicov"):
            return "rodinia_lc_gicov"
        if rest.startswith("lc_mgvf"):
            return "rodinia_lc_mgvf"

        algo = rest.split("_")[0]  # backprop/hotspot/kmeans/knn/dilate/lavaMD/pathfinder/streamcluster/...
        return f"rodinia_{algo}"

    # fallback: itself
    return s


# Ensure weight is float and add family field
ds_all = ds_all.map(lambda ex: {
    "weight": float(ex["weight"]),
    "family": family_id_from_kernel_name(ex["kernel_name"])
}, num_proc=4)


# Get list of kernels and families
kernel_names = sorted(set(ds_all["kernel_name"]))
families = sorted(set(ds_all["family"]))
print("Total kernels:", len(kernel_names))
print("Kernel names:", kernel_names)
print("Total families:", len(families))
print("Family names:", families)

# Set a kernel_weight 
counts = Counter(ds_all["kernel_name"])
N_total = len(ds_all)
K = len(counts)

alpha = 0.1 # smoothing power
kernel_weight = {}
for k, Nk in counts.items():
    kernel_weight[k] = (N_total / (K * Nk)) ** alpha

print(kernel_weight)

# train / test / val split
VAL_FAMS  = {"rodinia_pathfinder", "machsuite-sort-radix"}
TEST_FAMS = {"rodinia_knn"}
TRAIN_FAMS = set(families) - VAL_FAMS - TEST_FAMS

print("Train families ({}):".format(len(TRAIN_FAMS)), sorted(TRAIN_FAMS))
print("Val families   ({}):".format(len(VAL_FAMS)),  sorted(VAL_FAMS))
print("Test families  ({}):".format(len(TEST_FAMS)), sorted(TEST_FAMS))

train_fams = sorted(list(TRAIN_FAMS))
val_fams   = sorted(list(VAL_FAMS))
test_fams  = sorted(list(TEST_FAMS))

# Ensures that each split will include all the design points from FAMS respectively and nothing else
def in_family(example, names):
    return example["family"] in names

# Filter dataset 
ds_train = ds_all.filter(in_family, fn_kwargs={"names": train_fams}, num_proc=4)
ds_val   = ds_all.filter(in_family, fn_kwargs={"names": val_fams}, num_proc=4)
ds_test  = ds_all.filter(in_family, fn_kwargs={"names": test_fams}, num_proc=4)

ds_train = ds_train.add_column("row_id", list(range(len(ds_train))))
ds_val   = ds_val.add_column("row_id", list(range(len(ds_val))))
ds_test  = ds_test.add_column("row_id", list(range(len(ds_test))))

print(ds_train)
print(ds_val)
print(ds_test)

ds = DatasetDict(
    train=ds_train,
    validation=ds_val,
    test=ds_test,
    )


#-----------------------------------------------------------
#               Prompt Format And Tokenization
#-----------------------------------------------------------

MODEL_NAME = "deepseek-ai/deepseek-coder-7b-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


PROMPT_TEMPLATE = """
### Task
Given this HLS kernel with pragma placeholders of the form auto{{...}}. Output ONLY the filled directives, one per line, using EXACTLY the placeholders that appear in the kernel. The filled directives should be in the following format:

auto{{_PIPE_LX}} = <non-negative integer>
auto{{_UNROLL_LX}} = <non-negative integer>
auto{{_ARRAY_T_LX}} = <block|cyclic|complete>
auto{{_ARRAY_F_LX}} = <non-negative integer>
auto{{_ARRAY_D_LX}} = <non-negative integer>

Do not add explanations, comments, or extra headings.

### Kernel :
{code}

### Directives:
"""

# For every training example : full_text = prompt ("Task" + "Kernel") + target ("Directives")
def format_example(ex):
    prompt = PROMPT_TEMPLATE.format(code=ex["input"])
    full_text = prompt + ex["target"]
    return {"prompt": prompt, "full_text": full_text}


ds_all = ds_all.map(format_example, num_proc=4)
ds = ds.map(format_example, num_proc=4)


MAX_LENGTH = 6144

# We want to preserve the end of full_text (which contains all the directive lines)
# If the full_text tokens exceed the context window of the model then cut from the beginning of the prompt (no pragma placeholder from the "Kernel" field is missing)

def tokenize_and_mask(ex):
    prompt_ids_full = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(ex["target"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]	# eos_token shows to the model where to stop

    t_len = len(target_ids)

    if t_len >= MAX_LENGTH:	# no such case in our dataset
        kept_prompt_ids = []
        trunc_prompt_tokens = len(prompt_ids_full)
        prompt_len_kept = 0
        input_ids = target_ids[-MAX_LENGTH:]
    else:
        remaining_for_prompt = MAX_LENGTH - t_len
        trunc_prompt_tokens = max(0, len(prompt_ids_full) - remaining_for_prompt)
        kept_prompt_ids = prompt_ids_full[-remaining_for_prompt:] if trunc_prompt_tokens > 0 else prompt_ids_full
        prompt_len_kept = len(kept_prompt_ids)
        input_ids = kept_prompt_ids + target_ids

    attention_mask = [1] * len(input_ids)	# 1 --> real token => attend to it, 0 --> pad token

    labels = np.array(input_ids, dtype=np.int64)
    labels[:prompt_len_kept] = -100	# predict only the target tokens, not the prompt

    kernel_w = float(kernel_weight[ex["kernel_name"]])
    pareto_w = float(ex["weight"])
    ex_w = pareto_w * kernel_w

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "example_weight": ex_w,
        "length": len(input_ids),

        # ids / debug
        "row_id": ex["row_id"],
        "json_id": ex["json_id"],
        "kernel_name": ex["kernel_name"],
        "pareto_weight": pareto_w,
        "kernel_weight": kernel_w,

        # token stats
        "prompt_len_tokens": len(prompt_ids_full),
        "prompt_len_kept_tokens": prompt_len_kept,
        "target_len_tokens": t_len,
        "trunc_prompt_tokens": trunc_prompt_tokens,
    }


cols_to_remove = [c for c in ds["train"].column_names if c in ["input","target","weight","prompt","full_text"]] # remove raw fields, replace by tokenized

tokenized_ds = ds.map(
    tokenize_and_mask,
    remove_columns=cols_to_remove,
    desc="Tokenizing",
    )
tokenized_ds



#-----------------------------------------------------------
#                     Data Collator
#-----------------------------------------------------------

def data_collator(features):
    batch_inputs = {
        "input_ids": [f["input_ids"] for f in features],
        "attention_mask": [f["attention_mask"] for f in features],
    }

    batch = tokenizer.pad(
        batch_inputs,
        padding=True,            # pad to longest in batch
        return_tensors="pt",
        pad_to_multiple_of=8,    # good for FP16 kernels
    )

    max_len = batch["input_ids"].shape[1]
    labels = torch.full((len(features), max_len), -100, dtype=torch.long)
    for i, f in enumerate(features):
        l = len(f["labels"])
        labels[i, :l] = torch.tensor(f["labels"], dtype=torch.long)

    batch["labels"] = labels
    batch["example_weight"] = torch.tensor([float(f["example_weight"]) for f in features], dtype=torch.float32)

    batch["row_id"] = torch.tensor([f["row_id"] for f in features], dtype=torch.long)
    batch["pareto_weight"] = torch.tensor([f["pareto_weight"] for f in features], dtype=torch.float32)
    batch["kernel_weight"] = torch.tensor([f["kernel_weight"] for f in features], dtype=torch.float32)
    batch["prompt_len_tokens"] = torch.tensor([f["prompt_len_tokens"] for f in features], dtype=torch.long)
    batch["target_len_tokens"] = torch.tensor([f["target_len_tokens"] for f in features], dtype=torch.long)
    batch["trunc_prompt_tokens"] = torch.tensor([f["trunc_prompt_tokens"] for f in features], dtype=torch.long)
    batch["kernel_name"] = [f["kernel_name"] for f in features]  # keep as list of strings
    batch["json_id"] = torch.tensor([f["json_id"] for f in features], dtype=torch.long)
    batch["prompt_len_kept_tokens"] = torch.tensor([f["prompt_len_kept_tokens"] for f in features], dtype=torch.long)

    return batch



#-----------------------------------------------------------
#       Load Model (in 4-bit) And Attach QLoRA (with PEFT)
#-----------------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # reduces GPU memory
    bnb_4bit_use_double_quant=True, # double quantization --> further reduces memory with small quality loss
    bnb_4bit_quant_type="nf4",  # normal float 4 quantization --> preserves model quality better than naive 4-bit
    bnb_4bit_compute_dtype=torch.float16,
    # even though weights are 4-bit, computations are done in bfloat16 or float16 for numerical stability
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": GPU_ID},  # places the model layers on GPU
    trust_remote_code=True,
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

model = prepare_model_for_kbit_training(model)
# marks the right parts of the 4-bit model so gradients can flow through the LoRA adapters + enables gradient checkpointing (use_cache=False)

model.config.use_cache = False

# This keeps the 7B base model frozen in 4-bit and only train the small LoRA adapter matrices
lora_config = LoraConfig(
    r=16, # rank of the low-rank factors
    lora_alpha=32,  # scaling factor --> larger alpha => bigger LoRA influence, too large may destabilize training
    lora_dropout=0.05,  # adds regularization on the adapters to avoid overfitting
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    # typical LLaMA-style modules; Deepseek-coder is LLaMA-like
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # all attention projections (g, k, v, o) and feed-forward (MLP) projections (gate, up, down)
    )

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


#-----------------------------------------------------------
#              Wrapper for Weighted Fine Tuning
#-----------------------------------------------------------

# B = batch_size , T = sequence_length , V = vocab_size
# This wrapper ensures that pareto-near designs influence training more

class WeightedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        debug_keys = [
            "row_id", "json_id", "kernel_name",
            "pareto_weight", "kernel_weight",
            "prompt_len_tokens", "prompt_len_kept_tokens",
            "target_len_tokens", "trunc_prompt_tokens",
        ]
        debug_payload = {k: inputs.get(k, None) for k in debug_keys}

        # Keep these for diagnostics BEFORE popping
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        labels = inputs.pop("labels")
        w_raw = inputs.pop("example_weight").to(model.device)

        # remove non-model keys
        for k in debug_keys:
            if k in inputs:
                inputs.pop(k)

        outputs = model(**inputs)
        logits = outputs.logits  # (B,T,V)

        shift_logits = logits[:, :-1, :].contiguous()     # (B,T-1,V)
        shift_labels = labels[:, 1:].contiguous()         # (B,T-1)

        valid = shift_labels.ne(-100)                     # (B,T-1)
        mask = valid.float()

        vocab_size = shift_logits.size(-1)
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)

        tok_count = mask.sum(dim=1).clamp(min=1.0)
        loss_per_ex = (loss_per_token * mask).sum(dim=1) / tok_count

        # weights
        w_norm = w_raw / w_raw.mean().clamp(min=1e-8)
        w_clamped = w_norm.clamp(0.25, 4.0)

        loss = (loss_per_ex * w_clamped).sum() / w_clamped.sum().clamp(min=1e-8)

        with torch.no_grad():
            # teacher-forced token accuracy on valid target tokens
            pred_ids = shift_logits.argmax(dim=-1)
            tok_acc = (pred_ids[valid] == shift_labels[valid]).float().mean()
            tok_acc = tok_acc if torch.isfinite(tok_acc) else torch.tensor(0.0, device=tok_acc.device)

            # distribution stats
            ex_max = loss_per_ex.max()
            ex_p95 = torch.quantile(loss_per_ex, 0.95) if loss_per_ex.numel() > 1 else ex_max

            # truncation fraction in the batch (if present)
            trunc = debug_payload.get("trunc_prompt_tokens", None)
            if torch.is_tensor(trunc):
                trunc_frac = (trunc.to(loss.device) > 0).float().mean()
            else:
                trunc_frac = torch.tensor(0.0, device=loss.device)

            # log only on logging steps, only during training
            if model.training and (self.state.global_step % self.args.logging_steps == 0):
                self.log({
                    "loss_weighted": float(loss.detach().cpu()),
                    "loss_unweighted": float(loss_per_ex.mean().detach().cpu()),
                    "loss_ex_max": float(ex_max.detach().cpu()),
                    "loss_ex_p95": float(ex_p95.detach().cpu()),
                    "token_acc_tf": float(tok_acc.detach().cpu()),
                    "tok_mean": float(tok_count.mean().detach().cpu()),
                    "w_raw_min": float(w_raw.min().detach().cpu()),
                    "w_raw_max": float(w_raw.max().detach().cpu()),
                    "trunc_frac": float(trunc_frac.detach().cpu()),
                })

            # Spike-triggered heavy print (rare)
            spike = (not torch.isfinite(loss)) or (ex_max.item() > 8.0)  # threshold: tune (e.g., 6–10)
            if model.training and spike:
                i = int(torch.argmax(loss_per_ex).item())

                def _get_dbg(name, default="?"):
                    v = debug_payload.get(name, None)
                    if v is None: return default
                    if isinstance(v, list): return v[i]
                    if torch.is_tensor(v): return v[i].item() if v.ndim > 0 else v.item()
                    return v

                print("\n" + "="*110)
                print(f"[SPIKE] step={self.state.global_step} | i={i} | kernel={_get_dbg('kernel_name')} | json_id={_get_dbg('json_id')} | row_id={_get_dbg('row_id')}")
                print(f"loss_weighted={float(loss):.4f} | loss_ex={float(loss_per_ex[i]):.4f} | ex_max={float(ex_max):.4f} | tok_acc_tf={float(tok_acc):.3f}")
                print(f"tok_count={int(tok_count[i].item())} | w_raw={float(w_raw[i]):.4f} | w_clamped={float(w_clamped[i]):.4f} | trunc_prompt_tokens={_get_dbg('trunc_prompt_tokens')}")
                print("="*110)

                # fail fast on NaN/Inf so you don't waste GPU hours
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss detected (NaN/Inf). Aborting to protect training run.")


        return (loss, outputs) if return_outputs else loss




#-----------------------------------------------------------
#                       Training
#-----------------------------------------------------------

OUTPUT_DIR = "/home/elvouvali/deepseek_7B_qLoRA_all_kernels"

# os.environ["WANDB_WATCH"] = "all"

# wandb.login()

# ORIG_RUN_ID = "lm7c9kez"
# run = wandb.init(
#    project = "llm-for-hls",
#    id = ORIG_RUN_ID, # id of the interrupted run
#    resume = "must",  # resume the interrupted run
#    config = {
#      "learning_rate": 2e-4,
#      "architecture": "deepseek-ai/deepseek-coder-7b-base",
#      "dataset": "all_kernels_llm_data.jsonl",
#      "epochs": 3,
#    }
# )

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,  
    gradient_accumulation_steps=4,  # effective batch size = 4*2 = 8 (fits in memory + reasonably stable gradients)
    num_train_epochs=3,
    learning_rate=2e-4, # typical for LoRA
    weight_decay=0.0,
    warmup_ratio=0.03,  # 3% of total steps are LR warm-up => we avoid large gradients at the start
    lr_scheduler_type="cosine", # LR gently decays to near-zero
    optim = "paged_adamw_8bit",
    per_device_eval_batch_size=2,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps = 500,
    save_steps = 500,
    logging_steps = 20,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    group_by_length=True,
    length_column_name="length",

    fp16 = True,
    gradient_checkpointing=True,

    dataloader_num_workers=10,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    report_to="none",
    remove_unused_columns = False,
)


# training_args = TrainingArguments(
#    output_dir=OUTPUT_DIR,
#    per_device_train_batch_size=2,
#    gradient_accumulation_steps=4,
#    max_steps=5,
#    logging_steps=1,
#    save_strategy="no",
#    group_by_length=True,
#    length_column_name="length",
#    fp16=True,
#    remove_unused_columns=False,
#    label_names=["labels"],
#    learning_rate=2e-4
# )


trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)



# Resume from checkpoint
CHECKPOINT_PATH = "/home/elvouvali/deepseek_7B_qLoRA_all_kernels/checkpoint-5302"
if CHECKPOINT_PATH is not None and os.path.isdir(CHECKPOINT_PATH):
    print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
    trainer.train(resume_from_checkpoint=CHECKPOINT_PATH)
else:
    print(f"No checkpoint found at {CHECKPOINT_PATH}, starting from scratch.")
    trainer.train()


# Save only the LoRA adapter
trainer.model.save_pretrained(OUTPUT_DIR + "/lora_adapter")
tokenizer.save_pretrained(OUTPUT_DIR + "/tokenizer")

# wandb.finish()




