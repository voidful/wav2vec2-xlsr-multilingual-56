# wav2vec2-xlsr-multilingual-56

*56 language, 1 model Multilingual ASR*

Fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on 56 language using the [Common Voice](https://huggingface.co/datasets/common_voice).  
When using this model, make sure that your speech input is sampled at 16kHz.

For usage: [https://huggingface.co/voidful/wav2vec2-xlsr-multilingual-56](https://huggingface.co/voidful/wav2vec2-xlsr-multilingual-56)

## Data preprocessing

1. normalization:  
All normalization process packed in this repo: https://github.com/voidful/asrp  
For normalization in each language: https://github.com/voidful/asrp/blob/main/asrp/preprocessing.py  
The normalization function reference to https://huggingface.co/models?filter=xlsr-fine-tuning-week
   


