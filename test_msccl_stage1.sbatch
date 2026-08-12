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
#
# But mscclGetCaptureStatus logs at INFO under NCCL_INIT|NCCL_NET on every
# collective call, from the application thread - so it interleaves mid-line
# with the nccl-tests result table and makes it unreadable. NCCL_DEBUG_FILE
# routes all NCCL debug output to per-host/per-pid files instead, leaving
# stdout with nothing but the result table. %h and %p expand to hostname and
# pid; it is set per case inside run_case.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,NET

TESTBIN=$HOME/nccl-tests/build/all_gather_perf
mkdir -p logs/msccl_stage1

run_case() {
    local label="$1"
    local caselog="logs/msccl_stage1/${SLURM_JOB_ID}_${label}.log"
    local dbgdir="logs/msccl_stage1/debug_${SLURM_JOB_ID}_${label}"
    mkdir -p "$dbgdir"

    # Send NCCL's debug stream to files so it cannot interleave with the result
    # table on stdout. One file per host and pid.
    export NCCL_DEBUG_FILE="$dbgdir/nccl.%h.%p.log"

    echo ""
    echo "========================================================"
    echo "  CASE $label"
    echo "  NCCL_IB_FRAG_SIZE=$NCCL_IB_FRAG_SIZE  NCCL_IB_FRAG_BATCH=$NCCL_IB_FRAG_BATCH"
    echo "  $(date)"
    echo "========================================================"

    srun --mpi=pmix --ntasks=3 --ntasks-per-node=1 --export=ALL \
         $TESTBIN -b 3K -e 128M -f 2 -g 1 -c 1 -n 20 -w 5 -G 0 -z 0 2>&1 | tee "$caselog"

    echo "--- checks for CASE $label ---"

    # Assertions now read the debug files, not the result log.
    local dbg="$dbgdir"/*.log

    # 1. Did MSCCL actually run? NCCL_ALGO=MSCCL,RING,TREE falls back silently
    #    to RING/TREE when no algorithm matches nRanks, in which case this test
    #    says nothing about the MSCCL path.
    grep -h -m1 "Using MSCCL Algo files from" $dbg 2>/dev/null | sed 's/^/  MSCCL: /' \
        || echo "  !!! WARN: no 'Using MSCCL Algo files' line - may have fallen back to RING/TREE."
    grep -h -m1 "MSCCL: Setup Kernel finished" $dbg 2>/dev/null | sed 's/^/  MSCCL: /'

    # 2. Is fragmentation active?
    if grep -qh "MTU fragmentation on" $dbg 2>/dev/null; then
        grep -h -m1 "MTU fragmentation on" $dbg | sed 's/^/  ACTIVE: /'
    else
        echo "  ACTIVE: fragmentation NOT active (expected for case A)"
    fi

    # 3. Correctness, from the now-clean result table.
    grep -E "Out of bounds values|# Wrong|errors" "$caselog" | sed 's/^/  RESULT: /'

    # 4. Must never appear. On the MSCCL path nreqs==1 and sends are <= stepSize
    #    (~512kB, ~129 WRs against 2048 slots), so the budget cannot legitimately
    #    be exhausted - "posting unfragmented" here is a bug, not a tuning issue.
    for pat in "posting unfragmented" "credit ring overflow" "ibv_post_send() failed" "arena overflow"; do
        if grep -qh "$pat" $dbg 2>/dev/null; then
            echo "  !!! FAIL: '$pat' found in $dbgdir"
            grep -h -m3 "$pat" $dbg | sed 's/^/      /'
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
