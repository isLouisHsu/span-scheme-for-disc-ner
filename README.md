# A Simple but Effective Span-level Tagging Scheme for Discontinuous Named Entity Recognition

Official repository for the paper "A Simple but Effective Span-level Tagging Scheme for Discontinuous Named Entity Recognition" (https://link.springer.com/article/10.1007/s00521-024-09454-y).

## Prepare Data

**Download datasets**

- CADEC dataset can be downloaded at: https://data.csiro.au/dap/landingpage?pid=csiro:10948&v=3&d=true
- ShARe data can be downloaded at: https://physionet.org/

**Download Pretrained BERT Model**

- ClinicalBERT: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
- BioBERT: https://huggingface.co/dmis-lab/biobert-base-cased-v1.2

**Preprocessing**

CADEC:
```bash
bash scripts/build_cadec_data_for_discontinuous_ner.sh
```
NOTE: The processing pipeline follows https://github.com/daixiangau/acl2020-transition-discontinuous-ner.

## Run Experiments

```bash
python run_disc_ner.py \
    --experiment_code=exp_cadec_biobert_gcn_seed42 \
    --task_name=cadec \
    --model_type=bert \
    --do_lower_case \
    --pretrained_model_path=./pretrained/biobert-base-cased-v1.2 \
    --data_dir=data/processed/cadec/ \
    --train_input_file=train.txt \
    --eval_input_file=dev.txt \
    --test_input_file=test.txt \
    --output_dir=outputs/cadec/ \
    --do_train \
    --find_best_decode_thresh \
    --evaluate_during_training \
    --train_max_seq_length=256 \
    --eval_max_seq_length=512 \
    --test_max_seq_length=512 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=24 \
    --per_gpu_test_batch_size=24 \
    --gradient_accumulation_steps=1 \
    --learning_rate=2e-5 \
    --other_learning_rate=1e-3 \
    --loss_type='ce' \
    --weight_decay=0.0 \
    --max_grad_norm=1.0 \
    --scheduler_type=cosine \
    --num_train_epochs=12 \
    --checkpoint_mode=max \
    --checkpoint_save_best \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.6 \
    --negative_sampling=0.0 \
    --max_span_length=15 \
    --width_embedding_size=128 \
    --label_smoothing=0.0 \
    --decode_thresh=0.5 \
    --use_sinusoidal_width_embedding \
    --do_lstm \
    --num_lstm_layers=2 \
    --do_biaffine_rel \
    --use_last_n_layers=4 \
    --warmup_proportion=0.1 \
    --agg_last_n_layers=mean \
    --do_gcn \
    --gcn_hidden_size=128 \
    --seed=42
```

```bash
python run_disc_ner.py \
    --experiment_code=exp_share13_biobert_gcn_seed42 \
    --task_name=share13 \
    --model_type=bert \
    --pretrained_model_path=./pretrained/Bio_ClinicalBERT \
    --data_dir=data/processed/share13/ \
    --train_input_file=train.txt \
    --eval_input_file=dev.txt \
    --test_input_file=test.txt \
    --output_dir=outputs/share13/ \
    --do_train \
    --evaluate_during_training \
    --train_max_seq_length=200 \
    --eval_max_seq_length=512 \
    --test_max_seq_length=512 \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=8 \
    --per_gpu_test_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --learning_rate=2e-5 \
    --loss_type='ce' \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --max_grad_norm=1.0 \
    --scheduler_type=cosine \
    --num_train_epochs=12 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.5 \
    --negative_sampling=0.0 \
    --max_span_length=15 \
    --width_embedding_size=128 \
    --label_smoothing=0.0 \
    --decode_thresh=0.5 \
    --use_sinusoidal_width_embedding \
    --do_lstm \
    --num_lstm_layers=2 \
    --do_biaffine_rel \
    --use_last_n_layers=4 \
    --warmup_proportion=0.1 \
    --agg_last_n_layers=mean \
    --do_gcn \
    --gcn_hidden_size=128 \
    --seed=42
```

```bash
python run_disc_ner.py \
    --experiment_code=exp_share14_biobert_gcn_seed42 \
    --task_name=share14 \
    --model_type=bert \
    --pretrained_model_path=./pretrained/Bio_ClinicalBERT \
    --data_dir=data/processed/share14/ \
    --train_input_file=train.txt \
    --eval_input_file=dev.txt \
    --test_input_file=test.txt \
    --output_dir=outputs/share14/ \
    --do_train \
    --evaluate_during_training \
    --train_max_seq_length=200 \
    --eval_max_seq_length=512 \
    --test_max_seq_length=512 \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=8 \
    --per_gpu_test_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --learning_rate=2e-5 \
    --loss_type='ce' \
    --other_learning_rate=1e-3 \
    --weight_decay=0.01 \
    --max_grad_norm=1.0 \
    --scheduler_type=cosine \
    --num_train_epochs=12 \
    --checkpoint_mode=max \
    --checkpoint_monitor=eval_f1_micro_all_entity \
    --checkpoint_predict_code=checkpoint-eval_f1_micro_all_entity-best \
    --classifier_dropout=0.5 \
    --negative_sampling=0.0 \
    --max_span_length=15 \
    --width_embedding_size=128 \
    --label_smoothing=0.0 \
    --decode_thresh=0.5 \
    --use_sinusoidal_width_embedding \
    --do_lstm \
    --num_lstm_layers=2 \
    --do_biaffine_rel \
    --use_last_n_layers=4 \
    --warmup_proportion=0.1 \
    --agg_last_n_layers=mean \
    --do_gcn \
    --gcn_hidden_size=128 \
    --seed=42
```

