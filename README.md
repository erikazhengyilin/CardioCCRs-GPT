#### Exploring GPT's Diagnostic Capabilities for Cardiovascular Conditions with Clinical Case Reports  
   
Project Overview: This study evaluates the capability of the GPT API in diagnosing cardiovascular diseases (CVDs) from Metadata Acquired from Clinical Case Reports (MACCRs) 
  
For data preprocessing, run   
``` shell
python maccrs_preprocess.py
```
   
For diagnosis prediction, run  
``` shell
python main.py --model gpt_model_name --sample_mode sample_mode --sample_size
```  
For example, to use the GPT-4o-mini model and sampling 5 random CCRs for few-shot learning,  
``` shell
python main.py --model gpt-4o-mini --sample_mode random_k --sample_size 5
```  