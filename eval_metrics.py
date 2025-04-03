import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import evaluate  # using evaluate library

# Define a prompt template for clear separation between question and answer
PROMPT_TEMPLATE = """<|QUESTION|>: {question}
<|ANSWER|>:"""

def load_model_and_tokenizer(adapter_path, base_model_name="mistralai/Mistral-7B-v0.1"):
    # BitsAndBytes config for 4-bit quantization (should match training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    # Load the base model in 4-bit with device_map="auto"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    # Load the adapter (LoRA) on top of the base model
    model = PeftModel.from_pretrained(base_model, adapter_path)
    # Load the tokenizer from the adapter directory
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model.eval()
    return tokenizer, model

def evaluate_and_compute_rouge(adapter_path, test_file="data/qa/test.json"):
    tokenizer, model = load_model_and_tokenizer(adapter_path)
    
    # Load the test set
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    # Load the ROUGE metric using evaluate
    rouge_metric = evaluate.load("rouge")
    
    results = []
    for example in test_data:
        question = example["question"]
        reference = example["reference_answer"]
        
        prompt = PROMPT_TEMPLATE.format(question=question)
        encoding = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Compute ROUGE scores comparing generated text with the reference answer.
        rouge_result = rouge_metric.compute(predictions=[generated], references=[reference])
        # Directly use the float value from the result.
        rouge_l = rouge_result["rougeL"]
        
        # Decide if the answer is "close enough" (for example, threshold at 0.3)
        close_enough = rouge_l >= 0.3
        
        result = {
            "question": question,
            "generated_answer": generated,
            "reference_answer": reference,
            "rougeL": rouge_l,
            "close_enough": close_enough
        }
        results.append(result)
        print(f"Q: {question}")
        print(f"Generated: {generated}")
        print(f"Reference: {reference}")
        print(f"ROUGE-L: {rouge_l:.3f} -> {'Close enough' if close_enough else 'Not close enough'}\n")
    
    return results

if __name__ == "__main__":
    # Adapter path where your model and tokenizer are saved
    adapter_path = "mistral_construction_model"
    evaluate_and_compute_rouge(adapter_path)
