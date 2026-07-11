# config.py

DATA_DIR          = "plantvillage"   # update this to your actual path
NUM_CLIENTS       = 10
CLIENTS_PER_ROUND = 5
ROUNDS            = 10
LOCAL_EPOCHS      = 1
BATCH_SIZE        = 8   # Reduced from 32 to avoid OOM on limited memory systems
LR                = 1e-3
MU                = 0.01
TOP_K             = 0.3
ALPHA             = 0.1
RESULTS_PATH      = "results/metrics.csv"