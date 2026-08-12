#!/bin/bash
#SBATCH --job-name=nccl-stage1-frag
#SBATCH --partition=ja
#SBATCH --nodelist=ja2,ja3
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/nccl_stage1_%j.log

# Stage 1 (MTU fragmentation + chained writes) first correctness run.
#
# This is a CORRECTNESS test, not a benchmark: -c 1 is on, iteration counts are
# low, the size range stops at 512M, and only one node pair is used. Run the
# perf sweep from test.sbatch afterwards, once this is green.
#
# Four cases are run back to back so the comparison is within one job:
#   A  fragmentation off                     - baseline, must match stock
#   B  fragmentation off, deep send queue    - isolates the SQ resize
#   C  fragmentation on, k=1                 - simplest fragmented path
#   D  fragmentation on, k=2                 - the configuration of interest
#   E  fragmentation on, k=2, awkward sizes  - exercises the tail fragment

# --- Environment setup (mirrors test.sbatch) ---
source /opt/spack/share/spack/setup-env.sh
spack load /37dkg34   # OpenMPI
spack load /3nth2ba   # CUDA

NCCL_BUILD=$HOME/msccl_azure/executor/msccl-executor-nccl
export LD_LIBRARY_PATH=$(spack location -i /37dkg34)/lib:$(spack location -i /3nth2ba)/lib64:$NCCL_BUILD/build/lib/:$LD_LIBRARY_PATH

export PMIX_MCA_psec=native

# --- Pre-flight: confirm we are testing the branch we think we are ---
# The fragmentation code only exists on stage1-mtu-fragmentation, and
# LD_LIBRARY_PATH silently falls back to a system libnccl if the build is
# missing. Both failure modes look like "fragmentation did nothing".
echo "========================================================"
echo "  branch : $(git -C $NCCL_BUILD rev-parse --abbrev-ref HEAD)"
echo "  commit : $(git -C $NCCL_BUILD rev-parse --short HEAD)"
echo "  dirty  : $(git -C $NCCL_BUILD status --porcelain | wc -l) file(s) modified"
echo "  libnccl: $(ls -l $NCCL_BUILD/build/lib/libnccl.so.* 2>/dev/null | head -1)"
echo "========================================================"
if [[ ! -e $NCCL_BUILD/build/lib/libnccl.so ]]; then
    echo "FATAL: no build at $NCCL_BUILD/build/lib - run make first."
    exit 1
fi

# --- Network transport (unchanged from test.sbatch) ---
# GPU-NIC distance is SYS on ja[2-4] so GDR is off; data stages through pinned
# host memory. Fragmentation is indifferent to this - the SGEs just point at
# host rather than device memory - but it does mean bandwidth here is not
# representative of a GPUDirect setup.
export NCCL_IB_HCA=mlx5_0
export NCCL_NET_GDR_LEVEL=0

# --- Make sendrecv resemble the MSCCL collective path ---
# ncclSend/ncclRecv run at connIndex==1 with no graph, so net.cc gives them
# shared=1: shared buffer pool, NCCL_MODE_OFFSET indirection, comm reuse (hence
# nreqs>1), deferred credits. MSCCL is always shared=0. Forcing SHARED_BUFFERS=0
# restores all four to the MSCCL shape, including nreqs==1.
export NCCL_NET_SHARED_BUFFERS=0
# p2p sizes transfers with p2pChunkSize (128kB default), not the 512kB step
# MSCCL gets from buffSizes[SIMPLE]/NCCL_STEPS. Without this we would be testing
# 32-fragment sends instead of 128-fragment ones.
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_BUFFSIZE=4194304
export NCCL_PROTO=Simple

# --- Keep the QP layout trivial for the first run ---
# One pass per isend, one QP: no chunkSize/length divergence, no zero-length
# passes. Multi-QP is a later test.
export NCCL_IB_QPS_PER_CONNECTION=1
export NCCL_IB_SPLIT_DATA_ON_QPS=0

# mlx5_0 is InfiniBand, so comm->ar == 1 and stock already appends a separate
# 0-byte IMM WR above IB_AR_THRESHOLD (8192). Setting it to 0 makes stock do
# that at every size, so baseline and fragmented differ ONLY in fragmentation
# rather than also in WR shape.
export NCCL_IB_AR_THRESHOLD=0

# --- Debug logging ---
# NET is required: the "MTU fragmentation on: ..." line and all three failure
# warnings come from the net subsystem.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,NET

PAIR="ja2,ja3"
TESTBIN=$HOME/nccl-tests/build/sendrecv_perf
mkdir -p logs/stage1

run_case() {
    local label="$1"; shift
    local sizes="$1"; shift
    local caselog="logs/stage1/${SLURM_JOB_ID}_${label}.log"

    echo ""
    echo "========================================================"
    echo "  CASE $label"
    echo "  NCCL_IB_FRAG_SIZE=$NCCL_IB_FRAG_SIZE"
    echo "  NCCL_IB_FRAG_BATCH=$NCCL_IB_FRAG_BATCH"
    echo "  NCCL_IB_SEND_QUEUE_DEPTH=$NCCL_IB_SEND_QUEUE_DEPTH"
    echo "  sizes: $sizes"
    echo "  $(date)"
    echo "========================================================"

    # -c 1 : verify results. This is the point of the run.
    # -n/-w low: correctness, not timing.
    srun --mpi=pmix --ntasks=2 --ntasks-per-node=1 \
         --nodelist="$PAIR" --export=ALL \
         $TESTBIN $sizes -g 1 -c 1 -n 20 -w 5 -G 0 -z 0 2>&1 | tee "$caselog"

    # --- Post-run assertions on the log ---
    echo "--- checks for CASE $label ---"

    if grep -q "MTU fragmentation on" "$caselog"; then
        grep -m1 "MTU fragmentation on" "$caselog" | sed 's/^/  ACTIVE: /'
    else
        echo "  ACTIVE: fragmentation NOT active (expected for cases A and B)"
    fi

    # Wrong-result count from nccl-tests.
    grep -E "Out of bounds values|# Wrong|errors" "$caselog" | sed 's/^/  RESULT: /'

    # These three must never appear. "posting unfragmented" in particular is a
    # bug rather than a tuning issue here: 512kB at nreqs==1 is ~129 WRs against
    # 2048 slots, so the budget cannot legitimately be exhausted.
    for pat in "posting unfragmented" "credit ring overflow" "ibv_post_send() failed" "arena overflow"; do
        if grep -q "$pat" "$caselog"; then
            echo "  !!! FAIL: '$pat' found in $caselog"
            grep -m3 "$pat" "$caselog" | sed 's/^/      /'
        fi
    done
}

# A - baseline. With FRAG_SIZE=0 the send queue depth is unchanged at
#     2*MAX_REQUESTS, so this is genuinely stock behaviour.
export NCCL_IB_FRAG_SIZE=0
export NCCL_IB_FRAG_BATCH=2
export NCCL_IB_SEND_QUEUE_DEPTH=0
run_case "A_baseline" "-b 8 -e 512M -f 2"

# B - deep send queue, still no fragmentation. If A and B differ, the delta in
#     C/D is queue sizing rather than WR shape.
export NCCL_IB_SEND_QUEUE_DEPTH=2048
run_case "B_deepsq" "-b 8 -e 512M -f 2"

# C - smallest possible fragmented send first, then the full sweep. One
#     fragment plus the IMM WR: if this fails nothing else matters.
export NCCL_IB_FRAG_SIZE=-1
export NCCL_IB_FRAG_BATCH=1
export NCCL_IB_SEND_QUEUE_DEPTH=0
run_case "C_frag_k1_tiny" "-b 8 -e 8"
run_case "C_frag_k1" "-b 8 -e 512M -f 2"

# D - the configuration of interest.
export NCCL_IB_FRAG_BATCH=2
run_case "D_frag_k2" "-b 8 -e 512M -f 2"

# E - sizes that are not multiples of the 4kB fragment size. Powers of two never
#     exercise the short tail fragment; these do.
for s in 4095 4097 8191 12345 131071 524287; do
    run_case "E_odd_$s" "-b $s -e $s"
done

echo ""
echo "========================================================"
echo "  Stage 1 correctness run complete: $(date)"
echo "  Per-case logs: logs/stage1/${SLURM_JOB_ID}_*.log"
echo "========================================================"
