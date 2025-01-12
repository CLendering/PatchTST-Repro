if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/supervised" ]; then
    mkdir ./logs/supervised
fi

model_name=PatchTST
model_identifier=patchtst_etth2
dataset=etth2
input_length=336

for prediction_length in 96 192 336 720
do
    python -u src/training/train.py \
      --train_mode \
      --model_identifier $model_identifier'_'$input_length'_'$prediction_length \
      --model $model_name \
      --dataset $dataset \
      --features M \
      --input_length $input_length \
      --prediction_length $prediction_length \
      --encoder_input_size 7 \
      --num_encoder_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_fcn 128 \
      --dropout 0.3 \
      --fc_dropout 0.3 \
      --head_dropout 0 \
      --patch_length 16 \
      --stride 8 \
      --epochs 100 \
      --patience 20 \
      --bootstrap_iterations 1 --batch_size 128 --learning_rate 0.0001 >logs/supervised/$model_identifier'_'$input_length'_'$prediction_length.log 
done