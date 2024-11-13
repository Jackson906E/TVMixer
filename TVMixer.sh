export CUDA_VISIBLE_DEVICES=0

model_name=TVMixer

seq_len=24
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=16
batch_size=32

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path DMA_E.csv \
  --model_id DMA_E_24_1 \
  --model $model_name \
  --data custom \
  --target DMA_E \
  --features MS \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 1 \
  --e_layers $e_layers \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

