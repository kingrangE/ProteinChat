# 파일 이름: tune_script.py

import argparse
import torch
import re
from tqdm import tqdm
import yaml # PyYAML 설치 필요: pip install pyyaml

# W&B 라이브러리 임포트
import wandb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EvalPrediction
)
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model
from datasets import load_dataset

tokenizer = None

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
    
    # 80% train, 20% temp
    train_val_split = dataset.train_test_split(test_size=0.2, seed=42)
    val_test_split = train_val_split['test'].train_test_split(test_size=0.5, seed=42)
    
    train_data = train_val_split['train']
    validation_data = val_test_split['train']
    test_data = val_test_split['test']

    def preprocess(examples):
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

    train_dataset = train_data.map(preprocess, batched=True, remove_columns=train_data.column_names)
    validation_dataset = validation_data.map(preprocess, batched=True, remove_columns=validation_data.column_names)
    
    print(f"데이터 분할 완료: Train {len(train_dataset)}, Validation {len(validation_dataset)}, Test {len(test_data)}")
    return train_dataset, validation_dataset, test_data

def compute_metrics(p: EvalPrediction):
    preds_logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds_ids = preds_logits.argmax(axis=-1)
    
    labels_ids = p.label_ids
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    correct_predictions = 0
    for pred, label in zip(pred_str, label_str):
        true_answer_match = re.search(r'Answer:\s*(\d+)', label)
        predicted_answer_match = re.search(r'\d+', pred.split("Answer:")[-1])

        if true_answer_match and predicted_answer_match:
            try:
                if int(true_answer_match.group(1)) == int(predicted_answer_match.group(1)):
                    correct_predictions += 1
            except (ValueError, IndexError):
                continue
    
    accuracy = correct_predictions / len(labels_ids)
    return {"accuracy": accuracy}

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        global tokenizer 
        model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        train_dataset, validation_dataset, test_dataset = get_data(tokenizer)

        prompt_config = PromptTuningConfig(
            task_type="CAUSAL_LM",
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=config.num_virtual_tokens,
            prompt_tuning_init_text="Here are multiple choice questions about the law. Please choose the appropriate answer.",
            tokenizer_name_or_path=tokenizer.name_or_path,
        )
        peft_model = get_peft_model(model, prompt_config)
        peft_model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir=f'./results-{wandb.run.name}', 
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            fp16=torch.cuda.is_available(),
            evaluation_strategy="epoch",      
            save_strategy="epoch",            
            load_best_model_at_end=True,      
            metric_for_best_model="accuracy", 
            report_to="wandb",                
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            compute_metrics=compute_metrics, 
        )
        trainer.train()

        print("\n===== 최종 테스트셋으로 성능 평가 =====")
        final_metrics = trainer.evaluate(eval_dataset=test_dataset)
        wandb.log({"test_accuracy": final_metrics.get("eval_accuracy")})

        print("W&B Run 완료.")

if __name__ == "__main__":
    # 이 스크립트는 wandb agent에 의해 실행됩니다.
    # `wandb agent <SWEEP_ID>` 명령어를 사용하세요.
    train()
