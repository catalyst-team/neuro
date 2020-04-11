You can launch training by following command in terminal:
1. Download data via bash sripts/download_dataset.sh;
2. Prepare data by python3 scripts/prepare_data.py ;
3.1 Launch by command CUDA_VISIBLE_DEVICES=0 USE_APEX=0 catalyst-dl run --config=./config.yml.(if using 1 GPU);
3.2 CUDA_VISIBLE_DEVICES=0,2 USE_APEX=1 USE_DDP=1 catalyst-dl run --config=./config.yml --verbose ((if more using 1 GPU));
