import torch
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

def load_model_and_tokenizer(model_name="mistralai/Mistral-7B-v0.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Fix for no pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # Also make sure the model uses the correct pad token
    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model


class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.examples = []
        for item in raw_data:
            input_text = f"Context: {item['context']}\nQuestion: {item['question']}"
            target_text = f"{item['answer']}"
            input_ids = tokenizer.encode(input_text, truncation=True, max_length=max_length, padding="max_length")
            target_ids = tokenizer.encode(target_text, truncation=True, max_length=max_length, padding="max_length")

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
    train_dataset = QADataset("data/qa/train.json", tokenizer)

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        evaluation_strategy="no",
        fp16=False,  # Don't use mixed precision
        no_cuda=True  # Disable CUDA and MPS explicitly
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model("./mistral_construction_model")

if __name__ == "__main__":
    main()
