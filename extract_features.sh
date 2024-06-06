!/bin/bash
export pretrain_ds=audioset
tasks_dir=$1
output_dir=$2

CUDA_VISIBLE_DEVICES=0 python -m heareval.embeddings.runner hear_configs.ssast_tiny_200_16x4 --tasks-dir ${tasks_dir} --embeddings-dir ${output_dir}/${pretrain_ds}
CUDA_VISIBLE_DEVICES=0 python -m heareval.embeddings.runner hear_configs.mamba_ssast_tiny_200_16x4 --tasks-dir ${tasks_dir} --embeddings-dir ${output_dir}/${pretrain_ds}

python prepare_paths_v2.py --base_dir ${output_dir}/${pretrain_ds} --tgt_dir ${output_dir}/todo_${pretrain_ds}
