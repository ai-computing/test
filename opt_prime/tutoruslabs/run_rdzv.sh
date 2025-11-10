#!/bin/bash

export NCCL_DEBUG=ERROR
export TORCH_DIST_INIT_BARRIER=1

############################################
# User params
############################################
# models
# "meta-llama/Llama-3.2-1B"
# "meta-llama/Llama-3.2-3B"
# "meta-llama/Llama-3.1-8B-Instruct"
# "meta-llama/Llama-2-13b-chat-hf"
# "meta-llama/Llama-3.3-70B-Instruct"

# Required args
if [ $# -lt 4 ]; then
  echo "Usage: $0 <MODEL_NAME> <LLAMA_TOKEN> <NODE_RANK> <MASTER_ADDR> [NNODES] [NPROC_PER_NODE]"
  echo "Example: $0 meta-llama/Llama-3.1-8B-Instruct hf_xxxxx 0 10.0.0.11 8 8"
  exit 1
fi

MODEL_NAME="$1"
LLAMA_TOKEN="$2"
NODE_RANK="${3}"
MASTER_ADDR="${4}"
NNODES="${5:-8}"
NPROC_PER_NODE="${6:-8}"

WORLD_SIZE=$(( NNODES * NPROC_PER_NODE ))

BATCH_SIZES=(4096 2048 1024 512 256 128 64 32)
MICRO_BATCH_SIZES=(2048 1024 512 256 128 64 32 16 8 4)

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
MODEL_FILENAME=$(echo "$MODEL_NAME" | cut -d'/' -f2)
RESULT_FILEPATH="$RESULT_DIR/${MODEL_FILENAME}.csv"


# PP/TP/DP 조합 생성
COMBINATIONS=()
for ((PP=2; PP<=WORLD_SIZE; PP*=2)); do
  for ((TP=1; TP<=WORLD_SIZE; TP*=2)); do
    for ((DP=1; DP<=WORLD_SIZE; DP*=2)); do
      if [ $((PP * TP * DP)) -eq $WORLD_SIZE ]; then
        COMBINATIONS+=("$PP $TP $DP")
      fi
    done
  done
done
# 정렬: PP desc, TP desc, DP desc
#mapfile -t COMBINATIONS < <(
#  printf '%s\n' "${COMBINATIONS[@]}" | sort -k1,1nr -k2,2nr -k3,3nr
#)

echo "======== Generated PP/TP/DP combinations ========"
for COMBO in "${COMBINATIONS[@]}"; do
  read PP TP DP <<<"$COMBO"
  echo "PP=$PP, TP=$TP, DP=$DP"
done
echo "================================================="

COUNTER=0

# 모델 학습
for BATCH in "${BATCH_SIZES[@]}"; do
  for MICRO_BATCH in "${MICRO_BATCH_SIZES[@]}"; do

    if [ $MICRO_BATCH -ge $BATCH ]; then
      echo ">>> Skip: batch=$BATCH, micro_batch=$MICRO_BATCH (MICRO_BATCH >= BATCH)"
      continue
    fi

    for COMBO in "${COMBINATIONS[@]}"; do
      read PP TP DP <<<"$COMBO"
      
      RUN_ID="${MODEL_FILENAME}-${BATCH}-${MICRO_BATCH}-${PP}-${TP}-${DP}"
      
      COUNTER=$((COUNTER+1))
      RDZV_PORT=$((29500 + (COUNTER % 200)))

      RDZV_TIMEOUT=900

      echo "================================================="
      echo "RUN_ID            : $RUN_ID"
      echo "Model             : $MODEL_NAME"
      echo "Batch/Micro       : $BATCH / $MICRO_BATCH"
      echo "PP/TP/DP          : $PP / $TP / $DP"
      echo "Nodes x GPUs/node : $NNODES x $NPROC_PER_NODE (WORLD_SIZE=$WORLD_SIZE)"
      echo "RDZV              : c10d ${MASTER_ADDR}:${RDZV_PORT} (timeout=${RDZV_TIMEOUT}s)"
      echo "================================================="

      if [ "$NODE_RANK" -eq 0 ]; then
        ROLE="master"
      else
        ROLE="worker"
      fi
      echo ">>> This node is acting as: $ROLE"

      if [ "$ROLE" = "worker" ]; then
        echo "Waiting for master rendezvous (${MASTER_ADDR}:${RDZV_PORT})..."
        while ! nc -z "$MASTER_ADDR" "$RDZV_PORT"; do
          sleep 3
          echo "Still waiting for master..."
        done
        echo "Master is up, starting torchrun."
      fi

      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
        --nproc_per_node=$NPROC_PER_NODE \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${RDZV_PORT}" \
        --rdzv_id="${RUN_ID}" \
        --rdzv_conf="timeout=${RDZV_TIMEOUT}" \
        pp_train_llama.py \
          --llama_access_token "$LLAMA_TOKEN" \
          --model_name "$MODEL_NAME" \
          --batch_size $BATCH \
          --micro_batch_size $MICRO_BATCH \
          --pp_size $PP \
          --tp_size $TP \
          --dp_size $DP
      EXIT_CODE=$?


      echo ">>> Cleaning up GPU processes..."
      # Kill torchrun / python / CUDA processes that might hang
      pkill -9 -f "torchrun" || true
      pkill -9 -f "pp_train_llama.py" || true
      pkill -9 -f "python" || true
      sleep 3
      # Check if any GPU still in use, and free it
      fuser -v /dev/nvidia* -k 2>/dev/null || true
      echo ">>> GPU cleanup complete."


      # $? 변수로 종료 상태 코드 확인
      if [ $EXIT_CODE -eq 0 ]; then
        echo "SUCCESS: pp_train_llama.py completed successfully."
        echo "--- END ---"
        break 3
      else
        echo "FAILED (exit=$EXIT_CODE)"
      fi

      echo ">>> Done: batch=$BATCH, micro_batch=$MICRO_BATCH, pp=$PP, tp=$TP, dp=$DP"
      echo ""

      # 결과 파일에서 중복 라인 제거 (헤더 유지)
      ( head -n 1 "$RESULT_FILEPATH" && tail -n +2 "$RESULT_FILEPATH" | sort -u ) > "${RESULT_FILEPATH}.tmp" && mv "${RESULT_FILEPATH}.tmp" "$RESULT_FILEPATH"
      echo ">>> Duplicate lines removed from $RESULT_FILEPATH"

      sleep 10
    done
  done
done

# 결과 파일 정렬
(head -n 1 "$RESULT_FILEPATH" && tail -n +2 "$RESULT_FILEPATH" | sort -t',' -k1,1n -k2,2n) > "${RESULT_FILEPATH}.tmp" && mv "${RESULT_FILEPATH}.tmp" "$RESULT_FILEPATH"