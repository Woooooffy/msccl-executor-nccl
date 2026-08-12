#!/bin/bash
#SBATCH --job-name=msccl-stage1-frag
#SBATCH --partition=ja
#SBATCH --nodelist=ja2,ja3,ja4
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/msccl_stage1_%j.log

# Stage 1 (MTU fragmentation) correctness on the real MSCCL collective path.
#
# The sendrecv run (test_stage1.sbatch) had to force NCCL_NET_SHARED_BUFFERS=0
# and NCCL_P2P_NET_CHUNKSIZE to *imitate* MSCCL's transport shape, because p2p
# runs at connIndex==1 with shared=1. MSCCL is shared=0 natively, so none of
# that emulation is needed - and must NOT be carried over, or we would be
# testing a configuration nobody runs.
#
# Cases:
#   A  fragmentation off  - baseline
#   B  fragmentation on, k=2

# --- Environment setup ---
source /opt/spack/share/spack/setup-env.sh
spack load /pd6zjxg
spack load /3nth2ba
spack load /37dkg34

NCCL_BUILD=$HOME/msccl_azure/executor/msccl-executor-nccl
export LD_LIBRARY_PATH=$(spack location -i /37dkg34)/lib:$(spack location -i /3nth2ba)/lib64:$NCCL_BUILD/build/lib:$LD_LIBRARY_PATH

export PMIX_MCA_psec=native

# --- Pre-flight ---
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

# --- Network transport ---
# The original msccl-test script left the transport unpinned. Pin it to the
# same fabric the sendrecv stage-1 run used, or the two are not comparable:
# without NCCL_IB_HCA the job may pick a different HCA, and without
# NCCL_NET_GDR_LEVEL=0 the GDR decision could differ between runs.
export NCCL_IB_HCA=mlx5_0
export NCCL_NET_GDR_LEVEL=0

# --- MSCCL algorithm ---
export NCCL_ALGO=MSCCL,RING,TREE
# MSCCL_ALGO_DIR defaults to <libnccl dir>/msccl-algorithms. Uncomment to point
# somewhere explicit; the assertion below reports which directory was scanned.
# export MSCCL_ALGO_DIR=$HOME/msccl_azure/msccl-algorithms

# Match the QP layout of the sendrecv run so the comparison is like for like.
export NCCL_IB_QPS_PER_CONNECTION=1
export NCCL_IB_SPLIT_DATA_ON_QPS=0
# mlx5_0 is InfiniBand (ar==1), so stock already appends a separate 0-byte IMM
# WR above 8192 bytes. Forcing 0 makes it do so at every size, leaving
# fragmentation as the only difference between cases A and B.
export NCCL_IB_AR_THRESHOLD=0

# --- Debug logging ---
# NET is required for the fragmentation INFO line and the failure warnings.
# INIT is required to confirm MSCCL actually loaded an algorithm rather than
# silently falling back to RING/TREE.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,NET

TESTBIN=$HOME/nccl-tests/build/all_gather_perf
mkdir -p logs/msccl_stage1

run_case() {
    local label="$1"
    local caselog="logs/msccl_stage1/${SLURM_JOB_ID}_${label}.log"

    echo ""
    echo "========================================================"
    echo "  CASE $label"
    echo "  NCCL_IB_FRAG_SIZE=$NCCL_IB_FRAG_SIZE  NCCL_IB_FRAG_BATCH=$NCCL_IB_FRAG_BATCH"
    echo "  $(date)"
    echo "========================================================"

    srun --mpi=pmix --ntasks=3 --ntasks-per-node=1 --export=ALL \
         $TESTBIN -b 3K -e 128M -f 2 -g 1 -c 1 -n 20 -w 5 -G 0 -z 0 2>&1 | tee "$caselog"

    echo "--- checks for CASE $label ---"

    # 1. Did MSCCL actually run? NCCL_ALGO=MSCCL,RING,TREE falls back silently
    #    to RING/TREE when no algorithm matches nRanks, in which case this test
    #    says nothing about the MSCCL path.
    grep -m1 "algorithm directory" "$caselog" | sed 's/^/  ALGODIR: /'
    if grep -qE "MSCCL.*[Aa]lgo.*(load|select)|mscclSetupConnections" "$caselog"; then
        grep -mE3 "MSCCL.*[Aa]lgo.*(load|select)" "$caselog" | sed 's/^/  MSCCL: /'
    else
        echo "  !!! WARN: no MSCCL algorithm load line found - may have fallen back to RING/TREE."
        echo "      Check that an algorithm exists for nRanks=3 and this collective."
    fi

    # 2. Is fragmentation active?
    if grep -q "MTU fragmentation on" "$caselog"; then
        grep -m1 "MTU fragmentation on" "$caselog" | sed 's/^/  ACTIVE: /'
    else
        echo "  ACTIVE: fragmentation NOT active (expected for case A)"
    fi

    # 3. Correctness.
    grep -E "Out of bounds values|# Wrong|errors" "$caselog" | sed 's/^/  RESULT: /'

    # 4. Must never appear. On the MSCCL path nreqs==1 and sends are <= stepSize
    #    (~512kB, ~129 WRs against 2048 slots), so the budget cannot legitimately
    #    be exhausted - "posting unfragmented" here is a bug, not a tuning issue.
    for pat in "posting unfragmented" "credit ring overflow" "ibv_post_send() failed" "arena overflow"; do
        if grep -q "$pat" "$caselog"; then
            echo "  !!! FAIL: '$pat' found in $caselog"
            grep -m3 "$pat" "$caselog" | sed 's/^/      /'
        fi
    done
}

export NCCL_IB_FRAG_SIZE=0
export NCCL_IB_FRAG_BATCH=2
run_case "A_baseline"

export NCCL_IB_FRAG_SIZE=-1
run_case "B_frag_k2"

echo ""
echo "========================================================"
echo "  MSCCL stage 1 run complete: $(date)"
echo "  Per-case logs: logs/msccl_stage1/${SLURM_JOB_ID}_*.log"
echo "========================================================"
