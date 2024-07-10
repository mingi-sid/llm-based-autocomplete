import datetime
import json
import re
from glob import glob

import bitsandbytes as bnb
import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import (LoraConfig, PeftModel, TaskType, get_peft_model)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_json',   type=str, default=None, help='JSON file containing arguments. If None, use command line arguments. If arguments are provided in both, command line arguments will be used.')
    parser.add_argument('--model_id',    type=str, default='heegyu/kogpt-j-base', help='huggingface model id')
    parser.add_argument('--rank',        type=int, default=8, help='rank of lora')
    parser.add_argument('--lora_alpha',  type=int, default=16, help='alpha of lora')
    parser.add_argument('--max_length',  type=int, default=400, help='max length of input sequence')
    parser.add_argument('--batch_size',  type=int, default=1, help='batch size')
    parser.add_argument('--lr',          type=float, default=1e-6, help='learning rate')
    parser.add_argument('--num_epochs',  type=int, default=4, help='number of epochs')
    parser.add_argument('--device',      type=str, default='cuda', help='device')
    parser.add_argument('--cache_dir',   type=str, default='./model', help='Cache directory for saving model from huggingface')
    parser.add_argument('--load_path',   type=str, default=None, help='Directory path of lora weights and config for continued training. If None, initialize lora.')
    parser.add_argument('--data_glob',   type=str, default='./server/data/**/*.txt', help='Glob pattern for training data files.')
    parser.add_argument('--sample_by',   type=str, choices=['document', 'paragraph'], default='document', help='Sample by document or line.')
    parser.add_argument('--save_path',   type=str, default='./models', help='Directory path to save lora weights and configs.')
    parser.add_argument('--test_prompt', type=str, default='죽는 날까지 하늘을 우러러 한 점 부끄럼이 없기를, 잎새에 이는 바람에도 나는 괴로워했다.', help='Prompt for testing generation')

    args = parser.parse_args()
    if args.args_json is not None:
        try:
            with open(args.args_json) as f:
                args_json = json.load(f)
                for k, v in args_json.items():
                    if k not in args.__dict__:
                        setattr(args, k, v)
        except:
            print('Failed to load args_json. Using command line arguments.')

    return args

def main():
    args = get_args()

    model_id = args.model_id
    rank = args.rank
    lora_alpha = args.lora_alpha
    max_length = args.max_length
    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.num_epochs
    device = args.device if torch.cuda.is_available() else 'cpu'
    cache_dir = args.cache_dir
    load_path = args.load_path
    data_glob = args.data_glob
    save_path = args.save_path
    prompt = args.test_prompt

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)

    _ = model.eval()
    _ = model.to('cuda:0')

    is_saved_lora_available = True if (load_path is not None) else False
    if is_saved_lora_available:
        model = PeftModel.from_pretrained(model, load_path, is_trainable=True)
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)

    # Print example generation before fine-tuning
    with torch.no_grad():
        tokens = tokenizer(prompt, return_tensors='pt').to(device=device)
        gen_tokens = model.generate(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask'],
            do_sample=True,
            temperature=0.9,
            max_length=100,
            top_k=10)
        generated = tokenizer.batch_decode(gen_tokens)
    print('Test prompt: ', prompt)
    print('Generated text: ', generated[0])

    # Fine-tune LoRA
    _ = model.train()
    model.print_trainable_parameters()

    # Preparing Training Dataset
    text_files = glob(data_glob, recursive=True)
    print('Number of training files:', len(text_files))

    try:
        dataset = load_dataset('text', data_files=text_files,
                            sample_by='document', split='train')
    except ValueError:
        print('No training files found.')
        return

    def preprocess_function(examples):
        # Remove redundant newlines
        examples["text"] = re.sub(r"\n\s*\n", "\n", examples["text"])
        model_inputs = tokenizer(examples["text"], padding='longest',
                                 return_tensors="pt")
        return model_inputs

    processed_datasets = dataset.map(
        preprocess_function,
        num_proc=1,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
        remove_columns='text'
    )

    print(processed_datasets)

    def chunk_text(examples):
        # Chunk into max_length tokens, return as a list of examples
        result = {}
        for k, v in examples.items():
            value_chunks = []
            v = sum(v, [])
            for v2 in v:
                #print(len(v2))
                for i in range(0, len(v2), max_length):
                    value_chunks.append(v2[i:i+max_length])
                # Another examples, starting from half of max_length
                for i in range(max_length//2, len(v2), max_length):
                    value_chunks.append(v2[i:i+max_length])
            value_chunks = [x for x in value_chunks if len(x) == max_length]
            result[k] = value_chunks[:]
            result['labels'] = result['input_ids'].copy()
        #print(len(result['input_ids']))
        print(np.array(result['input_ids']).shape)
        return result

    chunked_dataset = processed_datasets.map(
        chunk_text,
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        desc="Chunking text",
    )
    print(chunked_dataset)
    print(max(len(x) for x in chunked_dataset['input_ids']))
    #print('chunked_dataset[0]:', chunked_dataset[0])

    assert np.array(chunked_dataset['input_ids']).shape == \
           np.array(chunked_dataset['labels']).shape
    assert np.array(chunked_dataset['input_ids']).shape == \
           np.array(chunked_dataset['attention_mask']).shape

    torch.cuda.empty_cache()

    training_args=transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            warmup_steps=100, 
            num_train_epochs=num_epochs,
            learning_rate=lr, 
            fp16=True,
            logging_steps=10, 
            optim='adafactor',
            output_dir='outputs'
        )

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=chunked_dataset,
        args=training_args,
        data_collator=data_collator
    )

    with torch.cuda.amp.autocast(): 
        trainer.train()

    # Print example generation after fine-tuning
    model.eval()

    with torch.no_grad():
        tokens = tokenizer(prompt, return_tensors='pt').to(device='cuda')
        gen_tokens = model.generate(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask'],
            do_sample=True,
            temperature=0.9,
            max_length=100,
            top_k=10,
            repetition_penalty=10.0)
        generated = tokenizer.batch_decode(gen_tokens)
        print(generated[0])

    # Save LoRA model with timestamp
    model.save_pretrained(f"models/{model_id.replace('/', '_')}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")

if __name__ == '__main__':
    main()