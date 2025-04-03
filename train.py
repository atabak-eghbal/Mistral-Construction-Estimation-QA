import torch
import json
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, TaskType, get_peft_model

def load_model_and_tokenizer(model_name="mistralai/Mistral-7B-v0.1"):

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. bitsandbytes config for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # 3. Load model in 4-bit with device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 4. Setup LoRA config (Hyperparams are examples, tune as needed)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # 5. Wrap the base model with LoRA adapters
    model = get_peft_model(model, lora_config)

    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model


class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.examples = []
        for item in raw_data:
            input_text = f"Context: {item['context']}\nQuestion: {item['question']}"
            target_text = item["answer"]

            input_ids = tokenizer.encode(
                input_text, truncation=True, max_length=max_length, padding="max_length"
            )
            target_ids = tokenizer.encode(
                target_text, truncation=True, max_length=max_length, padding="max_length"
            )

            self.examples.append({
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(target_ids)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def main():
    tokenizer, model = load_model_and_tokenizer()

    # create your dataset
    train_dataset = QADataset("data/qa/train.json", tokenizer)

    # standard training args
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        logging_steps=10,
        save_steps=50,
        save_total_limit=1,
        evaluation_strategy="no",
        fp16=True,
        torch_compile=False  # safer with accelerate
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # pass model + training args + dataset to Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    trainer.train()

    # saving the LoRA-adapter + base model config
    model_save_path = "mistral_construction_model"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # If you want just LoRA weights (PEFT style):
    # model.save_pretrained(model_save_path)


if __name__ == "__main__":
    main()
