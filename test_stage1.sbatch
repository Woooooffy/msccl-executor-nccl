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
# PHASE=correctness (cases A-E), single QP:
#   A  fragmentation off                     - baseline, must match stock
#   B  fragmentation off, deep send queue    - isolates the SQ resize
#   C  fragmentation on, k=1                 - simplest fragmented path
#   D  fragmentation on, k=2                 - the configuration of interest
#   E  fragmentation on, k=2, awkward sizes  - exercises the tail fragment
#
# PHASE=qp (cases F-G): does multi-QP hide the fragmentation cost?
#   The A-E run showed a 3-9% regression at 64kB-1MB that decays to zero above
#   ~2MB, i.e. it lives in the latency-dominated regime. F sweeps QP count in
#   both striping modes over exactly that band, with repeats, to see whether
#   more queues recover it. G is the small-size correctness case that only
#   multi-pass configurations can reach.
#
# Set PHASE=correctness, qp, or all. Default qp, since A-E already passed.
PHASE="${PHASE:-qp}"
REPS="${REPS:-3}"

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

# --- QP layout ---
# Defaults for the correctness phase: one pass per isend, one QP, so there is
# no chunkSize/length divergence and no zero-length passes. The qp phase
# overrides both per case.
#
# NOTE: both ranks read these independently from the environment - there is no
# negotiation - and ncclIbIrecv posts recv WRs using the same
# `split ? nqps : ndevs` formula the sender uses to post IMM WRs. A mismatch
# between ranks desynchronises the walk and hangs. --export=ALL keeps them in
# step; do not set them per-rank.
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
    echo "  NCCL_IB_QPS_PER_CONNECTION=$NCCL_IB_QPS_PER_CONNECTION"
    echo "  NCCL_IB_SPLIT_DATA_ON_QPS=$NCCL_IB_SPLIT_DATA_ON_QPS"
    echo "  sizes: $sizes  iters: $ITERS"
    echo "  $(date)"
    echo "========================================================"

    # -c 1 throughout: the qp phase is the first time multi-pass configurations
    # run at all, so it doubles as their correctness test.
    srun --mpi=pmix --ntasks=2 --ntasks-per-node=1 \
         --nodelist="$PAIR" --export=ALL \
         $TESTBIN $sizes -g 1 -c 1 $ITERS -G 0 -z 0 2>&1 | tee "$caselog"

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

ITERS="-n 20 -w 5"

if [[ "$PHASE" == "correctness" || "$PHASE" == "all" ]]; then

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
#     exercise the short tail fragment; these do. Note nccl-tests rounds to the
#     datatype boundary, so ask for byte counts that survive that rounding.
for s in 4095 4097 8191 12345 131071 524287; do
    run_case "E_odd_$s" "-b $s -e $s"
done

fi  # PHASE correctness

if [[ "$PHASE" == "qp" || "$PHASE" == "all" ]]; then

# ============================================================
# F - can more QPs hide the fragmentation cost?
#
# The A-E run put the regression at 64kB-1MB, decaying to nothing by ~4MB,
# i.e. it is a latency-regime cost, not a bandwidth-regime one. So sweep
# exactly that band.
#
# The two striping modes test different mechanisms, and only one of them
# plausibly helps:
#
#   SPLIT_DATA_ON_QPS=0  passes stay at ndevs (=1), so an isend still puts all
#                        its fragments on one QP - but *successive* isends
#                        round-robin onto different QPs. Tests whether the cost
#                        is consecutive steps queueing behind each other in one
#                        send queue.
#
#   SPLIT_DATA_ON_QPS=1  passes = nqps, so one isend is split across N QPs with
#                        ~1/N the fragments each. Tests whether the HCA can
#                        drain N shallower queues faster than one deep one.
#                        Note this ADDS work: N trailing IMM WRs instead of 1,
#                        and N doorbell batches minimum.
#
# Expect neither to fully recover it. The doorbell count is set by total
# fragments and k, and every doorbell is issued by the same single proxy
# progress thread regardless of how many QPs exist - so if the cost is CPU-side
# WR construction and post_send calls, more queues cannot help. If multi-QP
# *does* recover it, that is evidence the cost is HCA-side queue drain instead,
# which would change where stage 3's pacing has to live.
#
# Repeated REPS times because the A-E conclusion rested on single runs.
# ============================================================

ITERS="-n 50 -w 20"
QPBAND="-b 64K -e 2M -f 2"

for rep in $(seq 1 $REPS); do
  for qps in 1 2 4; do
    for split in 0 1; do
      # qps=1 makes the two modes identical; only run one of them.
      [[ $qps -eq 1 && $split -eq 1 ]] && continue

      export NCCL_IB_QPS_PER_CONNECTION=$qps
      export NCCL_IB_SPLIT_DATA_ON_QPS=$split

      export NCCL_IB_FRAG_SIZE=0
      export NCCL_IB_FRAG_BATCH=2
      run_case "F_r${rep}_qp${qps}_split${split}_fragoff" "$QPBAND"

      export NCCL_IB_FRAG_SIZE=-1
      run_case "F_r${rep}_qp${qps}_split${split}_fragon" "$QPBAND"
    done
  done
done

# G - multi-pass correctness at small sizes. This is the case the A-E run could
#     not reach: with nqps>1 and split mode, chunkSize rounds up to 128B, so a
#     100-byte message gives pass 0 all the data and pass 1 a negative length.
#     That pass must still post its trailing IMM WR or the receiver waits
#     forever on an unconsumed recv WR. A hang here, rather than wrong data, is
#     the expected failure signature.
ITERS="-n 20 -w 5"
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_IB_FRAG_SIZE=-1
export NCCL_IB_FRAG_BATCH=2
for qps in 2 4; do
    export NCCL_IB_QPS_PER_CONNECTION=$qps
    run_case "G_split_small_qp${qps}" "-b 8 -e 8K -f 2"
done

fi  # PHASE qp

echo ""
echo "========================================================"
echo "  Stage 1 run complete (PHASE=$PHASE): $(date)"
echo "  Per-case logs: logs/stage1/${SLURM_JOB_ID}_*.log"
echo "========================================================"
