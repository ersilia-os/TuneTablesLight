DATA_PATH="/home/xxxxx/tunetables_light/tunetables_light/data/openml__higgs__146606"
RESUME="/home/xxxxx/tunetables_light/tunetables_light/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
EPOCHS=31

# Learning rate, gradient aggregation (epochs, AdamW)
# batch size, tuned_prompt_size

python train_loop.py --data_path $DATA_PATH --resume $RESUME --epochs $EPOCHS --prompt_tuning \
--tuned_prompt_size 1000 --lr 0.01 --aggregate_k_gradients 1 --batch_size 4 --seed 100

python train_loop.py --data_path $DATA_PATH --resume $RESUME --epochs $EPOCHS --prompt_tuning \
--tuned_prompt_size 1000 --lr 0.2 --aggregate_k_gradients 1 --batch_size 4 --seed 100

python train_loop.py --data_path $DATA_PATH --resume $RESUME --epochs $EPOCHS --prompt_tuning \
--tuned_prompt_size 1000 --lr 0.1 --aggregate_k_gradients 2 --batch_size 4 --seed 100 

python train_loop.py --data_path $DATA_PATH --resume $RESUME --epochs $EPOCHS --prompt_tuning \
--tuned_prompt_size 1000 --lr 0.1 --aggregate_k_gradients 1 --batch_size 8 --seed 100 

python train_loop.py --data_path $DATA_PATH --resume $RESUME --epochs $EPOCHS --prompt_tuning \
--tuned_prompt_size 1000 --lr 0.1 --aggregate_k_gradients 1 --batch_size 2 --seed 100

python train_loop.py --data_path $DATA_PATH --resume $RESUME --epochs $EPOCHS --prompt_tuning \
--tuned_prompt_size 1000 --lr 0.1 --aggregate_k_gradients 1 --batch_size 4 --bptt 512 

python train_loop.py --data_path $DATA_PATH --resume $RESUME --epochs $EPOCHS --prompt_tuning \
--tuned_prompt_size 1000 --lr 0.1 --aggregate_k_gradients 1 --batch_size 4 --bptt 2048 

