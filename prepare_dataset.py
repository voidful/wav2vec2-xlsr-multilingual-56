import os
from pathlib import Path
import sys

import asrp
import torch
import torchaudio
import pyarrow.parquet as pq

import datasets
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from argument_classes import ModelArguments, DataTrainingArguments

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

print(f'Using {data_args.preprocessing_num_workers} threads')
torch.set_num_threads(data_args.preprocessing_num_workers)

train_datasets = []
eval_datasets = []
for lang in asrp.langs:
    try:
        train_dataset = datasets.load_dataset(
            "common_voice", lang, split=data_args.train_split_name
        )
        eval_dataset = datasets.load_dataset("common_voice", lang, split="test")
        preprocessing_sentence = getattr(asrp, 'fun_' + lang.replace("-", "_"))
        train_dataset = train_dataset.map(preprocessing_sentence, keep_in_memory=True,
                                          num_proc=data_args.preprocessing_num_workers)
        eval_dataset = eval_dataset.map(preprocessing_sentence, keep_in_memory=True,
                                        num_proc=data_args.preprocessing_num_workers)
        train_datasets.append(train_dataset)
        eval_datasets.append(eval_dataset)
    except:
        pass

train_dataset = datasets.concatenate_datasets(train_datasets)
eval_dataset = datasets.concatenate_datasets(eval_datasets)

tokenizer = Wav2Vec2CTCTokenizer(
    "vocab.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
)

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16_000, padding_value=0.0, do_normalize=True, return_attention_mask=True
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

resampler = torchaudio.transforms.Resample(48_000, 16_000)
resampled_data_dir = Path('./resample_file')
resampled_data_dir.mkdir(exist_ok=True)


def load_resample_save(f):
    f = Path(f)
    new_path = resampled_data_dir / f'{f.stem}_resampled16k.pt'
    if not new_path.exists():
        speech_array, sampling_rate = torchaudio.load(f)
        speech_array_resampled = resampler(speech_array)
        input_values = processor(speech_array_resampled, sampling_rate=16_000).input_values
        input_values = torch.from_numpy(input_values).float().flatten()
        torch.save(input_values, new_path)
        file_len = len(input_values.squeeze().tolist())
    else:
        try:
            file_len = len(torch.load(new_path).squeeze().tolist())
        except:
            speech_array, sampling_rate = torchaudio.load(f)
            speech_array_resampled = resampler(speech_array)
            input_values = processor(speech_array_resampled, sampling_rate=16_000).input_values
            input_values = torch.from_numpy(input_values).float().flatten()
            torch.save(input_values, new_path)
            file_len = len(input_values.squeeze().tolist())
    return str(new_path), file_len


print('load resample save')


def speech_file_to_array_fn(batch):
    path, length = load_resample_save(batch['path'])
    batch['path'] = path
    batch['voice_len'] = length
    batch["sampling_rate"] = 16_000
    batch["target_text"] = batch["sentence"]
    return batch


train_dataset = train_dataset.map(
    speech_file_to_array_fn,
    num_proc=data_args.preprocessing_num_workers,
    keep_in_memory=True,
)
eval_dataset = eval_dataset.map(
    speech_file_to_array_fn,
    num_proc=data_args.preprocessing_num_workers,
    keep_in_memory=True,
)


def tokenize_targets(batch):
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch


print('preparing dataset: train')
train_dataset = train_dataset.map(
    tokenize_targets,
    batch_size=training_args.per_device_train_batch_size,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
)
print('preparing dataset: eval')
eval_dataset = eval_dataset.map(
    tokenize_targets,
    batch_size=training_args.per_device_train_batch_size,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
)

pq.write_table(train_dataset.data.table, f'./multi.train.parquet')
pq.write_table(eval_dataset.data.table, f'./multi.eval.parquet')

processor.save_pretrained(training_args.output_dir)
