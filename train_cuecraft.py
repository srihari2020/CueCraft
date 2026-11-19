import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

MODEL_NAME = "empero-ai/Qwythos-9B-Claude-Mythos-5-1M"
MAX_SEQ_LENGTH = 1024

# 1. Load model in 4-bit with automatic layer offloading for 8 GB VRAM
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
    offload_layers="auto",
)

# 2. Attach QLoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

SYSTEM_PROMPT = (
    "You are CueCraft, an expert multi-modal prompt engineer. "
    "Generate strict, zero-drift, production-grade prompts for [TEXT], [IMAGE], [VIDEO], and [MOVIE] tasks."
)

# 3. Format dataset using the model's chat template
def format_prompts(examples):
    texts = []
    for inst, out in zip(examples["instruction"], examples["output"]):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": inst},
            {"role": "assistant", "content": out},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )
    return {"text": texts}

dataset = load_dataset("json", data_files="data/cuecraft_train.jsonl", split="train")
dataset = dataset.map(format_prompts, batched=True)

# 4. Train using modern TRL SFTConfig (tuned for 8 GB VRAM)
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="outputs/cuecraft-lora",
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # Switch to num_train_epochs=3 once you have 500+ examples
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
    ),
)

trainer.train()

# 5. Save LoRA adapter & export Q4_K_M GGUF
model.save_pretrained("CueCraft-LoRA")
tokenizer.save_pretrained("CueCraft-LoRA")

model.save_pretrained_gguf(
    "CueCraft-GGUF",
    tokenizer,
    quantization_method="q4_k_m",
)

print("Done! GGUF exported to CueCraft-GGUF/")
