from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import argparse
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import re
from tqdm import tqdm
import pandas as pd

def model_test(model, tokenizer):
    prompt = "Explain who you are"

    messages = [
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    output = model.generate(
        input_ids.to(model.device),
        max_new_tokens=128,
        do_sample=False,
    )
    print(tokenizer.decode(output[0]))

def get_model_info(model):
    print(model)

def create_mmlu_prompt(question, choices):
    prompt = f"""다음은 객관식 질문입니다. 네 가지 선택지 중 가장 적절한 답의 번호(0, 1, 2, 3)를 고르세요.

        Question: {question}
        Choices:
        0: {choices[0]}
        1: {choices[1]}
        2: {choices[2]}
        3: {choices[3]}

        Answer: """
    return prompt

def get_data(tokenizer):
    dataset = load_dataset("cais/mmlu", "professional_law", split="test")
    split_dataset = dataset.train_test_split(test_size=0.5, seed=42)
    train_data, test_data = split_dataset['train'],split_dataset['test']
    def preprocess_for_training(examples):
        prompts = []
        for q, c, a in zip(examples['question'], examples['choices'], examples['answer']):
            prompt = create_mmlu_prompt(q, c) + str(a)
            prompts.append(prompt)
        tokenized_inputs = tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        return tokenized_inputs

    train_dataset = train_data.map(preprocess_for_training, batched=True, remove_columns=train_data.column_names)
    
    print(f"Split is done : Train {len(train_dataset)}, Test {len(test_data)}")
    return train_dataset, test_data

def evaluate_model(model, tokenizer, test_dataset):
    model.eval()
    correct_predictions = 0
    total_predictions = len(test_dataset)
    
    with torch.no_grad():
        for item in tqdm(test_dataset, desc="Evaluating"):
            question = item['question']
            choices = item['choices']
            true_answer = item['answer']

            prompt = create_mmlu_prompt(question, choices)
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                pad_token_id=tokenizer.eos_token_id
            )
            response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
            match = re.search(r'\d+', response_text)
            if match:
                try:
                    predicted_answer = int(match.group(0))
                    if predicted_answer == true_answer:
                        correct_predictions += 1
                except (ValueError, IndexError):
                    continue
        
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

def lora_finetuning(model, tokenizer, train_dataset):
    lora_config = LoraConfig(
        r=16,  
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],  
        lora_dropout=0.05,
        task_type="CAUSAL_LM" 
    )
    use_fp16 = torch.cuda.is_available()
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir='./results',          
        num_train_epochs=1,              
        per_device_train_batch_size=4,   # Batchsize
        gradient_accumulation_steps=4,   # Gradient 4번 누적 후 update (Batchsize 16와 같아짐)
        warmup_steps=10,                 # Warmup step 수
        weight_decay=0.01,               # 가중치 decay 비율
        logging_dir= './logs',
        logging_steps=10,                
        fp16=use_fp16,                   # CUDA가 가능할때만 가능하므로 use_fp16 전달
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    
    print("Finetuning is done")
    return peft_model

def get_model_and_tokenizer():
    model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

if __name__ =="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device =",device)
    parser  = argparse.ArgumentParser(description="LoRA Baseline Test")
    parser.add_argument("--test", action=argparse.BooleanOptionalAction, help="To make sure the model is working well")
    parser.add_argument("--info", action=argparse.BooleanOptionalAction, help="To visualize the model structure")
    parser.add_argument("--basemodel_metric", action=argparse.BooleanOptionalAction, help="To measure the performance of the base model")
    parser.add_argument("--lora_metric", action=argparse.BooleanOptionalAction, help="To measure the performance of the LoRA model")

    args = parser.parse_args() 

    model, tokenizer = get_model_and_tokenizer()
    train_dataset, test_dataset = get_data(tokenizer)

    if args.test :
        print("\n===== Base Model 실행 테스트 =====")
        model_test(model,tokenizer) # To make sure the model is working well
    if args.info :
        print("\n===== Base Model Info =====")
        get_model_info(model,tokenizer) # To visualize the model structure
    if args.basemodel_metric :
        print("\n===== Base Model 성능 측정 =====")
        base_model_score = evaluate_model(model, tokenizer, test_dataset)
        print(f"  - Base Model Accuracy: {base_model_score:.4f}")
    if args.lora_metric:
        print("\n===== LoRA Fine-tuning 진행 =====")
        tuned_model = lora_finetuning(model, tokenizer, train_dataset)

        print("\n===== Fine-tuned Model 성능 측정 =====")
        tuned_model_score = evaluate_model(tuned_model, tokenizer, test_dataset)
        print(f"  - LoRA Model Accuracy: {tuned_model_score:.4f}")
    
    if args.basemodel_metric and args.lora_metric:
        print("\n===== 최종 성능 비교 =====")
        print(f"Base Model Accuracy: {base_model_score:.4f}")
        print(f"LoRA Tuned Model Accuracy: {tuned_model_score:.4f}")
        improvement = tuned_model_score - base_model_score
        print(f"성능 향상: +{improvement:.4f}")