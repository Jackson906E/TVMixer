export CUDA_VISIBLE_DEVICES=1

model_name=PatchTST

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path DMA_E.csv \
  --model_id DMA_E_72_24 \
  --model $model_name \
  --data custom \
  --target DMA_E \
  --features MS \
  --seq_len 24 \
  --label_len 24 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --batch_size 32 \
  --train_epochs 10 \
  --d_ff 2048 \
  --d_model 1024 \
  --n_heads 8 \
  --itr 1 