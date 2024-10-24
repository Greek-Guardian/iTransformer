# export CUDA_VISIBLE_DEVICES=0

# 定义模型名称列表
model_names=("iTransformer" "iInformer" "iReformer" "iFlowformer" "iFlashformer" "Informer" "Reformer" "Flowformer" "Flashformer")

# 遍历模型名称
for model_name in "${model_names[@]}"
do
  # 遍历 model_structure 参数的值
  for i in {0..6}
  do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --model_id ECL_512_512 \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 512 \
      --pred_len 512 \
      --e_layers 2 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --itr 1 \
      --train_epochs 30 \
      --patience 5 \
      --use_multi_gpu \
      --devices 0,1 \
      --model_structure $i
  done
done