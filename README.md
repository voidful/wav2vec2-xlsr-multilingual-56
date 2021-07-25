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
| Common Voice Languages | Num. of data | Hour   | WER    | CER   |
|------------------------|--------------|--------|--------|-------|
| ar                     | 21744        | 81.5   | 75.29  | 31.23 |
| as                     | 394          | 1.1    | 95.37  | 46.05 |
| br                     | 4777         | 7.4    | 93.79  | 41.16 |
| ca                     | 301308       | 692.8  | 24.80  | 10.39 |
| cnh                    | 1563         | 2.4    | 68.11  | 23.10 |
| cs                     | 9773         | 39.5   | 67.86  | 12.57 |
| cv                     | 1749         | 5.9    | 95.43  | 34.03 |
| cy                     | 11615        | 106.7  | 67.03  | 23.97 |
| de                     | 262113       | 822.8  | 27.03  | 6.50  |
| dv                     | 4757         | 18.6   | 92.16  | 30.15 |
| el                     | 3717         | 11.1   | 94.48  | 58.67 |
| en                     | 580501       | 1763.6 | 34.87  | 14.84 |
| eo                     | 28574        | 162.3  | 37.77  | 6.23  |
| es                     | 176902       | 337.7  | 19.63  | 5.41  |
| et                     | 5473         | 35.9   | 86.87  | 20.79 |
| eu                     | 12677        | 90.2   | 44.80  | 7.32  |
| fa                     | 12806        | 290.6  | 53.81  | 15.09 |
| fi                     | 875          | 2.6    | 93.78  | 27.57 |
| fr                     | 314745       | 664.1  | 33.16  | 13.94 |
| fy-NL                  | 6717         | 27.2   | 72.54  | 26.58 |
| ga-IE                  | 1038         | 3.5    | 92.57  | 51.02 |
| hi                     | 292          | 2.0    | 90.95  | 57.43 |
| hsb                    | 980          | 2.3    | 89.44  | 27.19 |
| hu                     | 4782         | 9.3    | 97.15  | 36.75 |
| ia                     | 5078         | 10.4   | 52.00  | 11.35 |
| id                     | 3965         | 9.9    | 82.50  | 22.82 |
| it                     | 70943        | 178.0  | 39.09  | 8.72  |
| ja                     | 1308         | 8.2    | 99.21  | 62.06 |
| ka                     | 1585         | 4.0    | 90.53  | 18.57 |
| ky                     | 3466         | 12.2   | 76.53  | 19.80 |
| lg                     | 1634         | 17.1   | 98.95  | 43.84 |
| lt                     | 1175         | 3.9    | 92.61  | 26.81 |
| lv                     | 4554         | 6.3    | 90.34  | 30.81 |
| mn                     | 4020         | 11.6   | 82.68  | 30.14 |
| mt                     | 3552         | 7.8    | 84.18  | 22.96 |
| nl                     | 14398        | 71.8   | 57.18  | 19.01 |
| or                     | 517          | 0.9    | 90.93  | 27.34 |
| pa-IN                  | 255          | 0.8    | 87.95  | 42.03 |
| pl                     | 12621        | 112.0  | 56.14  | 12.06 |
| pt                     | 11106        | 61.3   | 53.24  | 16.32 |
| rm-sursilv             | 2589         | 5.9    | 78.17  | 23.31 |
| rm-vallader            | 931          | 2.3    | 73.67  | 21.76 |
| ro                     | 4257         | 8.7    | 83.84  | 21.95 |
| ru                     | 23444        | 119.1  | 61.83  | 15.18 |
| sah                    | 1847         | 4.4    | 94.38  | 38.46 |
| sl                     | 2594         | 6.7    | 84.21  | 20.54 |
| sv-SE                  | 4350         | 20.8   | 83.68  | 30.79 |
| ta                     | 3788         | 18.4   | 84.19  | 21.60 |
| th                     | 4839         | 11.7   | 141.87 | 37.16 |
| tr                     | 3478         | 22.3   | 66.77  | 15.55 |
| tt                     | 13338        | 26.7   | 86.80  | 33.57 |
| uk                     | 7271         | 39.4   | 70.23  | 14.34 |
| vi                     | 421          | 1.7    | 96.06  | 66.25 |
| zh-CN                  | 27284        | 58.7   | 89.67  | 23.96 |
| zh-HK                  | 12678        | 92.1   | 81.77  | 18.82 |
| zh-TW                  | 6402         | 56.6   | 85.08  | 29.07 |


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

