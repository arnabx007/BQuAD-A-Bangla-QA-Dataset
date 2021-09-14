# BQuAD - A Bangla Question Answering Dataset

For the final year thesis project we had to develop an extractive Question-Answering system for educational purposes. We chose to go with the language model approach. At that time (Q4 2019) there were only 2 language model publicly available: [Multilingual-BERT (mBERT)](https://github.com/google-research/bert/blob/master/multilingual.md) & [Bangla-Electra](https://huggingface.co/monsoon-nlp/bangla-electra). 

To serve our purpose well, we trained a BERT model in Bangla, BERT-Bangla which is a medium sized BERT model (8 attention layers compared to the 12 layers in general BERT base models). Also there wasn't any Bangla QA dataset publicly available. Hence, we developed a Bangla reading comprehension dataset BQuAD (following the pattern of SQuAD v1.1). 

The paper **'An Approach to Extractive Bangla Question Answering Based On BERT-Bangla And BQuAD'** is available on ieexplore. 
https://ieeexplore.ieee.org/document/9528178

## BQuAD

In response to the need for a well-suited Bangla question-answering dataset, in early 2020, we developed BQuAD (Bangla Question Answering Dataset) comprising about 13,000 question-answer pairs and contexts featuring various topics from multiple domains stored in SQuAD v1.1 like json format. 

**Example:** 
```
[{'context': 'ঢাকা দক্ষিণ এশিয়ার রাষ্ট্র বাংলাদেশের রাজধানী ও বৃহত্তম শহর। প্রশাসনিকভাবে এটি দেশটির ঢাকা বিভাগের প্রধান শহর। ভৌগোলিকভাবে এটি বাংলাদেশের মধ্যভাগে বুড়িগঙ্গা নদীর উত্তর তীরে একটি সমতল এলাকাতে অবস্থিত। ঢাকা একটি অতিমহানগরী (মেগাশহর); ঢাকা মহানগরী এলাকার জনসংখ্যা প্রায় ১ কোটি ৫০ লক্ষ। জনসংখ্যার বিচারে এটি দক্ষিণ এশিয়ার চতুর্থ বৃহত্তম শহর (দিল্লি, করাচি ও মুম্বইয়ের পরেই) এবং সমগ্র বিশ্বের নবম বৃহত্তম শহর। জনঘনত্বের বিচারে ঢাকা বিশ্বের সবচেয়ে ঘনবসতিপূর্ণ মহানগরী; ১৩৪ বর্গমাইল আয়তনের এই শহরে প্রতি বর্গমাইল এলাকায় ১ লক্ষ ১৫ হাজার লোকের বাস।',
  
  'qas': 
  
  [{'answers': [{'answer_start': 263, 'text': 'প্রায় ১ কোটি ৫০ লক্ষ'}],
    'id': 'bquad--fgce-1011580',
    'question': 'বাংলাদেশের রাজধানী ঢাকা শহরের বর্তমান (২০১৯) জনসংখ্যা কত ?',
    'is_impossible': False},
   
   {'question': 'ভৌগোলিকভাবে ঢাকা কোথায় অবস্থিত?',
    'is_impossible': False,
    'id': 'bquad--fgce-1011581',
    'answers': [{'text': 'বাংলাদেশের মধ্যভাগে বুড়িগঙ্গা নদীর উত্তর তীরে',
      'answer_start': 128}]},
      
   {'question': 'ঢাকা শহরেরে আয়তন কত?',
    'is_impossible': False,
    'id': 'bquad--fgce-1011582',
    'answers': [{'text': '১৩৪ বর্গমাইল', 'answer_start': 475}]},
    
   {'question': 'দক্ষিণ এশিয়ার প্রথম বৃহত্তম শহর কোনগুলো?',
    'is_impossible': False,
    'id': 'bquad--fgce-1011583',
    'answers': [{'text': 'দিল্লি, করাচি ও মুম্বইয', 'answer_start': 345}]},
```  
  
[BQuAD is available here.](https://github.com/arnabx007/BQuAD-A-Bangla-QA-Dataset/blob/master/bquad.json)
 
## BERT-Bangla

We also pretrained and released a medium sized BERT model in Bangla on about 10GB of Bangla text data. We are releasing it with the benchmark scores comparing with other concurrent exisiting models: mBERT and Bangla Electra. The model was pretrained with the following configurations:

* Vocab Size: 75000
* No. of Parameters: 65 Million
* No. of Attention Heads: 8
* No. of Layers: 8
* Hidden Vector Size: 512
* Hidden Intermediate Size: 2048

The model was pre-trained on a cloud v3-8 TPU (128GB of memory) for 35 hours for a total of 1M steps with a batch size of 256. 90% of the total training steps were done using a maximum sequence length of 128 and the remaining 10% using a maximum sequence length of 512. The training scripts were taken from the 
[original BERT repository.](https://github.com/google-research/bert)


Download the BERT-Bangla torch model from [here](https://drive.google.com/file/d/1otZvEbA5WyZkyQ4e9ZaSopXzPPnVvM7d/view?usp=sharing) and place it in the **bert-bangla-model** directory alongside it's config.json and vocab.txt file.

### Benchmark scores: 
Model         |	Sentiment Analysis | Hate SpeechV2	| News Topic Classification	 | Average Accuracy
------------- | ------------------ | -------------  | -------------------------  | ----------------
mBERT  | 68.1% | 50.9% | 72.3% | 63.7%
Bangla Electra  | 69.2% | 34.3% | 82.3% | 61.9%
**BERT-Bangla**  | **69.7%** | **67.5%**| **84.5%** | **73.9%**

The sentiment analysis and hate speech detection datasets were taken from here: https://github.com/rezacsedu/Classification_Benchmarks_Benglai_NLP

For the news topic classification task, a publicly available dataset was used taken from here: 

The BERT-Bangla model had achieved an Exact Match accuracy of 45% and F1 accuracy of 78.5% on the BQuAD Dataset.


## Citation:

```
@INPROCEEDINGS{9528178,  
author={Saha, Arnab and Noor, Mirza Ifat and Fahim, Shahriar and Sarker, Subrata and Badal, Faisal and Das, Sajal},  
booktitle={2021 International Conference on Automation, Control and Mechatronics for Industry 4.0 (ACMI)},   
title={An Approach to Extractive Bangla Question Answering Based On BERT-Bangla And BQuAD},   
year={2021},  
volume={},  
number={},  
pages={1-6},  
doi={10.1109/ACMI53878.2021.9528178}}
```







