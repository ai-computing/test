#!/bin/bash
set -euo pipefail

############################################
# User params
############################################
# models
# "meta-llama/Llama-3.2-1B"
# "meta-llama/Llama-3.2-3B"
# "meta-llama/Llama-3.1-8B-Instruct"
# "meta-llama/Llama-2-13b-chat-hf"
# "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

# Required args
if [ $# -lt 3 ]; then
  echo "Usage: $0 <LLAMA_TOKEN> <NODE_RANK> <MASTER_ADDR> [NNODES] [NPROC_PER_NODE] [RDZV_PORT]"
  echo "Example: $0 hf_xxxxx 0 10.0.0.11 8 8 29501"
  exit 1
fi
LLAMA_TOKEN="$1"
NODE_RANK="${2}"
MASTER_ADDR="${3}"
NNODES="${4:-8}"
NPROC_PER_NODE="${5:-8}"
RDZV_PORT="${6:-29501}"

############################################
# Derived params
############################################
WORLD_SIZE=$(( NNODES * NPROC_PER_NODE ))

BATCH_SIZES=(32 64 128 256 512 1024 2048 4096)
MICRO_BATCH_SIZES=(4 8 16 32 64 128 256 512 1024 2048)

RESULT_DIR="results"
mkdir -p "$RESULT_DIR"
MODEL_FILENAME=$(echo "$MODEL_NAME" | cut -d'/' -f2)
RESULT_FILEPATH="$RESULT_DIR/${MODEL_FILENAME}.csv"

############################################
# NCCL / network sanity (optional but helpful)
############################################
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
# 환경에 따라 조정:
# export NCCL_IB_DISABLE=1        # IB 없거나 문제 시
# export NCCL_SOCKET_IFNAME=eth0  # 올바른 NIC로 지정

############################################
# Generate PP/TP/DP combinations (PP*TP*DP == WORLD_SIZE)
############################################
COMBINATIONS=()
# PP,TP,DP는 2의 거듭제곱 조합만 시도 (원래 스크립트 관례 유지)
pow2s=()
v=1
while [ $v -le $WORLD_SIZE ]; do pow2s+=($v); v=$((v*2)); done

for PP in "${pow2s[@]}"; do
  for TP in "${pow2s[@]}"; do
    for DP in "${pow2s[@]}"; do
      if [ $((PP * TP * DP)) -eq $WORLD_SIZE ]; then
        COMBINATIONS+=("$PP $TP $DP")
      fi
    done
  done
done

echo "======== Generated PP/TP/DP combinations (WORLD_SIZE=$WORLD_SIZE) ========"
for COMBO in "${COMBINATIONS[@]}"; do
  read PP TP DP <<<"$COMBO"
  echo "PP=$PP, TP=$TP, DP=$DP"
done
echo "=========================================================================="

############################################
# Helpers
############################################
cleanup() {
  echo "[NODE $NODE_RANK] Caught signal, cleaning up..."
  pkill -P $$ || true
}
trap cleanup INT TERM

have_nc=0
if command -v nc >/dev/null 2>&1; then have_nc=1; fi

wait_master_tcp() {
  local host="$1" port="$2"
  # rendezvous는 rank0가 먼저 떠야 포트가 열립니다.
  # 순서가 바뀌어도 아래 retry loop가 복구하므로 이 체크는 선택 사항입니다.
  # 다만 네트워크가 죽어있을 때 빨리 감지하려고 가벼운 ping만 합니다.
  if ping -c 1 -W 1 "$host" >/dev/null 2>&1; then
    echo "[NODE $NODE_RANK] master host reachable: $host"
  else
    echo "[NODE $NODE_RANK] WARN: cannot ping $host (will rely on elastic retry)"
  fi
}

dedup_csv_if_exists() {
  if [ -f "$RESULT_FILEPATH" ]; then
    ( head -n 1 "$RESULT_FILEPATH" && tail -n +2 "$RESULT_FILEPATH" | sort -u ) > "${RESULT_FILEPATH}.tmp" && mv "${RESULT_FILEPATH}.tmp" "$RESULT_FILEPATH"
    echo ">>> Deduped: $RESULT_FILEPATH"
  fi
}

############################################
# Main loop
############################################
wait_master_tcp "$MASTER_ADDR" "$RDZV_PORT"

MAX_RETRY=5
RDZV_TIMEOUT=900   # seconds

for COMBO in "${COMBINATIONS[@]}"; do
  read PP TP DP <<<"$COMBO"

  for BATCH in "${BATCH_SIZES[@]}"; do
    for MICRO_BATCH in "${MICRO_BATCH_SIZES[@]}"; do

      if [ $MICRO_BATCH -ge $BATCH ]; then
        echo ">>> Skip: batch=$BATCH, micro_batch=$MICRO_BATCH (MICRO>=BATCH)"
        continue
      fi

      RUN_ID="${MODEL_FILENAME}-${BATCH}-${MICRO_BATCH}-${PP}-${TP}-${DP}"
      echo "================================================="
      echo "RUN_ID            : $RUN_ID"
      echo "Model             : $MODEL_NAME"
      echo "Batch/Micro       : $BATCH / $MICRO_BATCH"
      echo "PP/TP/DP          : $PP / $TP / $DP"
      echo "Nodes x GPUs/node : $NNODES x $NPROC_PER_NODE (WORLD_SIZE=$WORLD_SIZE)"
      echo "RDZV              : c10d ${MASTER_ADDR}:${RDZV_PORT} (timeout=${RDZV_TIMEOUT}s)"
      echo "================================================="

      attempt=1
      while [ $attempt -le $MAX_RETRY ]; do
        echo "[RUN_ID=$RUN_ID][NODE $NODE_RANK] Attempt $attempt/$MAX_RETRY"

        # 약간의 랜덤 백오프로 초기 충돌 감소
        sleep $((RANDOM % 3))

        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE-1))) torchrun \
          --nproc_per_node="${NPROC_PER_NODE}" \
          --nnodes="${NNODES}" \
          --node_rank="${NODE_RANK}" \
          --rdzv_backend=c10d \
          --rdzv_endpoint="${MASTER_ADDR}:${RDZV_PORT}" \
          --rdzv_id="${RUN_ID}" \
          --rdzv_timeout="${RDZV_TIMEOUT}" \
          pp_train_llama.py \
            --llama_access_token "$LLAMA_TOKEN" \
            --model_name "$MODEL_NAME" \
            --batch_size $BATCH \
            --micro_batch_size $MICRO_BATCH \
            --pp_size $PP \
            --tp_size $TP \
            --dp_size $DP

        exit_code=$?
        if [ $exit_code -eq 0 ]; then
          echo "[RUN_ID=$RUN_ID] SUCCESS"
          break
        else
          echo "[RUN_ID=$RUN_ID] FAIL(exit=$exit_code). Backoff & retry..."
          sleep $((RANDOM % 10 + 5))
        fi
        attempt=$((attempt+1))
      done

      if [ $attempt -gt $MAX_RETRY ]; then
        echo "[RUN_ID=$RUN_ID] GAVE UP after $MAX_RETRY attempts."
      fi

      echo ">>> Done: $RUN_ID"
      dedup_csv_if_exists
      sleep 5
    done
  done
done

# 최종 정렬 (있을 때만)
if [ -f "$RESULT_FILEPATH" ]; then
  (head -n 1 "$RESULT_FILEPATH" && tail -n +2 "$RESULT_FILEPATH" | sort -t',' -k1,1n -k2,2n) > "${RESULT_FILEPATH}.tmp" && mv "${RESULT_FILEPATH}.tmp" "$RESULT_FILEPATH"
  echo ">>> Final sort: $RESULT_FILEPATH"
fi

echo "=== ALL DONE (NODE $NODE_RANK) ==="
