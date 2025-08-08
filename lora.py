"""
LORA based fine tuning of Counterspeech generation model, for given hatespeeach and style.
"""
import torch
import pandas as pd
import os

from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline
)
from trl import SFTTrainer

# --- Configuration ---
BASE_MODEL_PATH = "/home/models/Llama-3.1-8B"
OUTPUT_MODEL_DIR = "Llama-3.1-8B-finetune"
# Add path of the training, evaluation, and test datasets
# Ensure these paths are correct and accessible
TRAIN_PATH = '<path>'
EVAL_PATH = '<path>'
TEST_PATH = '<path>'
OUTPUT_FILE_PATH = "<output path>"
BATCH_SIZE = 8

# --- Data Processing ---
def make_prompt(data):
    style = data['Style']
    hs = data['Hatespeech']
    return f"{hs}\n{style}\n"

def process_train(data_df):
    data_df = data_df.dropna(subset=["Hatespeech", "Style", "Counterspeech"])
    data_df = data_df[
        (data_df["Hatespeech"].str.strip() != "") &
        (data_df["Style"].str.strip() != "") &
        (data_df["Counterspeech"].str.strip() != "")
    ]
    data_df['prompt'] = data_df[['Hatespeech', 'Style']].apply(make_prompt, axis=1)
    data_df["text"] = data_df[["prompt", "Counterspeech"]].apply(
        lambda x: f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{x['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{x['Counterspeech']}<|eot_id|>",
        axis=1
    )
    return data_df

def process_test(data_df):
    data_df = data_df.dropna(subset=["Hatespeech", "Style"])
    data_df = data_df[
        (data_df["Hatespeech"].str.strip() != "") &
        (data_df["Style"].str.strip() != "")
    ]
    data_df['prompt'] = data_df[['Hatespeech', 'Style']].apply(make_prompt, axis=1)
    data_df["text"] = data_df["prompt"].apply(
        lambda p: f"user\n\n{p}<|eot_id|>assistant\n\n"
    )
    return data_df

# --- Model Loading ---
def get_model_and_tokenizer(model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer

# --- Training ---
def train():
    model, tokenizer = get_model_and_tokenizer(BASE_MODEL_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    eval_df = pd.read_csv(EVAL_PATH)
    train_data = Dataset.from_pandas(process_train(train_df))
    eval_data = Dataset.from_pandas(process_train(eval_df))
    peft_config = LoraConfig(
        r=2,
        lora_alpha=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    training_arguments = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        learning_rate=5e-5,
        save_strategy="no",
        logging_steps=10,
        num_train_epochs=1,
        fp16=True,
        max_grad_norm=0.0,
        group_by_length=False,
        evaluation_strategy="epoch",
        push_to_hub=False
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        args=training_arguments,
        tokenizer=tokenizer,
        dataset_text_field='text',
        max_seq_length=150,
        packing=False,
    )
    trainer.train()
    trainer.save_model(OUTPUT_MODEL_DIR)
    print("Training complete and model saved.")

# --- Inference ---
def inference():
    # Load and prepare test data
    test_df = pd.read_csv(TEST_PATH)
    test_df = test_df[:10]
    processed_df = process_test(test_df.copy())
    print(f"Loaded and processed {len(processed_df)} test samples.")

    # Load fine-tuned model and tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, OUTPUT_MODEL_DIR)
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    prompts = processed_df["text"].tolist()
    outputs = pipe(
        prompts,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False,
        temperature=None,
        top_p=None,
        return_full_text=False,
        batch_size=BATCH_SIZE,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_responses = [seq[0]['generated_text'] for seq in outputs]
    results_df = pd.DataFrame({
        "Hatespeech": processed_df["Hatespeech"],
        "Style": processed_df["Style"],
        "Ground_Truth_CS": processed_df.get("Counterspeech", ""),
        "CS": generated_responses
    })
    # results_df['CS'] = results_df['CS'].str.replace("<|eot_id|>", "").str.strip()
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    results_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"Results saved to: {OUTPUT_FILE_PATH}")
    print(results_df.head())

# --- Main ---
if __name__ == "__main__":
    train()
    inference()
