import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def evaluate(adapter_path, base_model_name="mistralai/Mistral-7B-v0.1", questions=None):
    if questions is None:
        questions = [
            "How does TaksoAi ensure accurate fitting measurements?",
            "What file formats does the software export to?",
            "What does cloud-based collaboration offer?"
        ]
    
    # Configure bitsandbytes for 4-bit quantization (same as during training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load the base model in 4-bit with device_map auto
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load the adapter (LoRA) on top of the base model
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Load the tokenizer from the adapter directory (which includes updated config)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    model.eval()
    
    for q in questions:
        # Tokenize the input question and move to the model's device
        input_ids = tokenizer.encode(q, return_tensors="pt").to(model.device)
        outputs = model.generate(
            input_ids, 
            max_length=100,
            do_sample=True, 
            temperature=0.7
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Q: {q}\nA: {answer}\n")

if __name__ == "__main__":
    test_questions = [
        "How does TaksoAi ensure accurate fitting measurements?",
        "What file formats does the software export to?",
        "What does cloud-based collaboration offer?"
    ]
    # 'mistral_construction_model' is the directory where your adapter was saved.
    evaluate("mistral_construction_model", questions=test_questions)
