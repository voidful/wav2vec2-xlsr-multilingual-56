# wav2vec2-xlsr-multilingual-56

*56 language, 1 model Multilingual ASR*

Fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on 56 language using the [Common Voice](https://huggingface.co/datasets/common_voice).  
When using this model, make sure that your speech input is sampled at 16kHz.

For usage: [https://huggingface.co/voidful/wav2vec2-xlsr-multilingual-56](https://huggingface.co/voidful/wav2vec2-xlsr-multilingual-56)

## THX
All language preprocess comes from `xlsr-fine-tuning-week`  
Data prepare and Model training script modified from https://github.com/maxidl/wav2vec2   
Thank you all participants for the ideas/experiments/opinions to make it happen.   

## Result
| Common Voice Languages | Num. of data | Hour   | CER   |
|------------------------|--------------|--------|-------|
| ar                     | 21744        | 81.5   | 31.27 |
| as                     | 394          | 1.1    | 46.03 |
| br                     | 4777         | 7.4    | 41.14 |
| ca                     | 301308       | 692.8  | 10.39 |
| cnh                    | 1563         | 2.4    | 23.11 |
| cs                     | 9773         | 39.5   | 12.57 |
| cv                     | 1749         | 5.9    | 34.01 |
| cy                     | 11615        | 106.7  | 23.93 |
| de                     | 262113       | 822.8  | 6.51  |
| dv                     | 4757         | 18.6   | 30.18 |
| el                     | 3717         | 11.1   | 58.69 |
| en                     | 580501       | 1763.6 | 14.84 |
| eo                     | 28574        | 162.3  | 6.23  |
| es                     | 176902       | 337.7  | 5.42  |
| et                     | 5473         | 35.9   | 20.80 |
| eu                     | 12677        | 90.2   | 7.32  |
| fa                     | 12806        | 290.6  | 15.09 |
| fi                     | 875          | 2.6    | 27.60 |
| fr                     | 314745       | 664.1  | 13.94 |
| fy-NL                  | 6717         | 27.2   | 26.58 |
| ga-IE                  | 1038         | 3.5    | 50.98 |
| hi                     | 292          | 2.0    | 57.34 |
| hsb                    | 980          | 2.3    | 27.18 |
| hu                     | 4782         | 9.3    | 36.74 |
| ia                     | 5078         | 10.4   | 11.37 |
| id                     | 3965         | 9.9    | 22.82 |
| it                     | 70943        | 178.0  | 8.72  |
| ja                     | 1308         | 8.2    | 61.91 |
| ka                     | 1585         | 4.0    | 18.57 |
| ky                     | 3466         | 12.2   | 19.83 |
| lg                     | 1634         | 17.1   | 43.84 |
| lt                     | 1175         | 3.9    | 26.82 |
| lv                     | 4554         | 6.3    | 30.79 |
| mn                     | 4020         | 11.6   | 30.15 |
| mt                     | 3552         | 7.8    | 22.94 |
| nl                     | 14398        | 71.8   | 19.01 |
| or                     | 517          | 0.9    | 27.42 |
| pa-IN                  | 255          | 0.8    | 42.00 |
| pl                     | 12621        | 112.0  | 12.07 |
| pt                     | 11106        | 61.3   | 16.33 |
| rm-sursilv             | 2589         | 5.9    | 23.30 |
| rm-vallader            | 931          | 2.3    | 21.70 |
| ro                     | 4257         | 8.7    | 21.93 |
| ru                     | 23444        | 119.1  | 15.18 |
| sah                    | 1847         | 4.4    | 38.47 |
| sl                     | 2594         | 6.7    | 20.52 |
| sv-SE                  | 4350         | 20.8   | 30.78 |
| ta                     | 3788         | 18.4   | 21.60 |
| th                     | 4839         | 11.7   | 37.24 |
| tr                     | 3478         | 22.3   | 15.55 |
| tt                     | 13338        | 26.7   | 33.59 |
| uk                     | 7271         | 39.4   | 14.35 |
| vi                     | 421          | 1.7    | 66.31 |
| zh-CN                  | 27284        | 58.7   | 23.94 |
| zh-HK                  | 12678        | 92.1   | 18.82 |
| zh-TW                  | 6402         | 56.6   | 29.08 |


## Data preprocessing

1. normalization:  
All normalization process packed in this repo: https://github.com/voidful/asrp  
For normalization in each language: https://github.com/voidful/asrp/blob/main/asrp/preprocessing.py  
The normalization function reference to https://huggingface.co/models?filter=xlsr-fine-tuning-week  
Welcome to modify and contribute, to make normalization results better.
   
2. get vocabulary list
I download all xlsr model's vocabulary from huggingface. Merge them in character level and remove duplicates.
   
3. run `prepare_dataset.py` for preprocessing.

## Training
`run_finetuning.py` for model training, here is my training setting:

```shell
python -m torch.distributed.launch --nproc_per_node 2 \
  run_finetuning.py \
  --model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
  --preprocessing_num_workers="16" \
  --overwrite_output_dir \
  --num_train_epochs="20" \
  --per_device_train_batch_size="9" \
  --gradient_accumulation_steps="5" \
  --learning_rate="1e-5" \
  --warmup_steps="500" \
  --evaluation_strategy="no" \
  --logging_steps="10" \
  --logging_first_step=True \
  --gradient_checkpointing=False \
  --save_total_limit="10" \
  --freeze_feature_extractor \
  --attention_dropout="0.00" \
  --hidden_dropout="0.00" \
  --feat_proj_dropout="0.0" \
  --mask_time_prob="0.05" \
  --layerdrop="0.00" \
  --save_strategy "epoch" \
  --do_train \
  --fp16 \
  --output_dir "./wav2vec2-xlsr-multilingual-56" \
  --report_to "wandb" \
  --dataloader_num_workers="16" \
  --group_by_length
```

