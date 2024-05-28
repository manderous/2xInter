# 2xInter
This project includes the code implemented with PyTorch and the paper 'Prototype-based Prompt-Instance Interaction with Causal Intervention for Few-shot Event Detection' published at The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024).

https://aclanthology.org/2024.lrec-main.1161.pdf

Run the model:

 ```
python fsl_bart_contra.py --dataset ace -n 5 -k 5 --encoder bart_contra_multipos_neg --model proto_bart_contra --bert_pretrained facebook/bart-base
 ```
