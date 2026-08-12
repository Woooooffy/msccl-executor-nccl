#!/bin/bash
#SBATCH --job-name=nccl-pairwise
#SBATCH --partition=ja
#SBATCH --nodelist=ja2,ja3,ja4
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/nccl_pairwise_%j.log

# --- Environment setup (mirrors msccl-test script) ---
source /opt/spack/share/spack/setup-env.sh

# OpenMPI 
spack load /37dkg34

# CUDA
spack load /3nth2ba

# NCCL/MSCCL library paths (same as msccl-test script).
export LD_LIBRARY_PATH=$(spack location -i /37dkg34)/lib:$(spack location -i /3nth2ba)/lib64:$HOME/msccl_azure/executor/msccl-executor-nccl/build/lib/:$LD_LIBRARY_PATH

# --- PMIx security ---
# Avoids munge authentication failure on this cluster.
export PMIX_MCA_psec=native

# --- Network transport ---
# IB is confirmed working (mlx5_0, HDR 200 Gb/s) from perftest benchmarks.
# GPU-NIC distance is SYS (cross-NUMA) on ja[2-4], so GPUDirect RDMA is
# disabled - NCCL will stage data through pinned host memory on both send
# and receive sides. This is the same path msccl used.
export NCCL_IB_HCA=mlx5_0
export NCCL_NET_GDR_LEVEL=0

# --- Debug logging ---
# INIT,ENV confirms which transport and GDR path NCCL actually selected.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV

# --- rdmatop RDMA-bandwidth monitoring ---
# One rdmatop capture per node, running as an overlapping job step so it
# doesn't consume any of the job's allocated GPU/CPU resources.
#
# rdmatop is a full-screen TUI, so it needs a real pty. `script`(1) can't
# provide one here: it derives the pty from an existing controlling
# terminal, and a batch job session has none at all - not just a non-tty
# stdout. So instead we run rdmatop inside a detached `tmux` session,
# which allocates its own pty independent of any controlling terminal,
# and periodically snapshot the rendered pane (plain text) to a log file.
#
# sbatch runs a non-interactive, non-login shell, so PATH may not include
# ~/.cargo/bin even if rdmatop is installed there. Rather than rely on
# PATH, point straight at the known binary and verify it's executable.
RDMATOP_BIN="$HOME/.cargo/bin/rdmatop"
if [[ ! -x "$RDMATOP_BIN" ]]; then
    RDMATOP_BIN=$(command -v rdmatop || true)
fi
TMUX_BIN=$(command -v tmux || true)
MONITOR_NODES=(ja2 ja3 ja4)
MONITOR_PIDS=()
MONITOR_SNAPSHOT_INTERVAL=2  # seconds between pane snapshots

start_rdmatop_monitors() {
    if [[ -z "$RDMATOP_BIN" ]]; then
        echo "WARNING: rdmatop not found in PATH - skipping RDMA monitoring."
        echo "         Install with 'cargo install rdmatop' or the crazyguitar PPA."
        return
    fi
    if [[ -z "$TMUX_BIN" ]]; then
        echo "WARNING: tmux not found in PATH - skipping RDMA monitoring."
        echo "         tmux is required to give rdmatop a pty inside a batch job."
        return
    fi

    mkdir -p logs/rdmatop

    for NODE in "${MONITOR_NODES[@]}"; do
        LOGFILE="logs/rdmatop/${NODE}_${SLURM_JOB_ID}.log"
        STDOUTFILE="logs/rdmatop/${NODE}_${SLURM_JOB_ID}.stdout"
        SESSION="rdmatop_${SLURM_JOB_ID}"
        SOCKET="rdmatop_${SLURM_JOB_ID}"

        srun --overlap \
             --nodes=1 \
             --ntasks=1 \
             --nodelist="$NODE" \
             bash -c "
                 export LANG=C.UTF-8 LC_ALL=C.UTF-8
                 # Use a job-scoped socket (-L) so we always get a fresh tmux
                 # server with the right locale, rather than possibly
                 # attaching to a pre-existing server on this node that
                 # started under a non-UTF-8 locale (tmux fixes its locale
                 # at server startup and won't pick up LANG afterward).
                 tmux -L '$SOCKET' kill-server 2>/dev/null
                 tmux -L '$SOCKET' new-session -d -s '$SESSION' '$RDMATOP_BIN'
                 sleep 2
                 while tmux -L '$SOCKET' has-session -t '$SESSION' 2>/dev/null; do
                     {
                         echo \"----- \$(date '+%Y-%m-%d %H:%M:%S') -----\"
                         tmux -L '$SOCKET' capture-pane -p -t '$SESSION'
                     } >> '$LOGFILE'
                     sleep $MONITOR_SNAPSHOT_INTERVAL
                 done
             " > "$STDOUTFILE" 2>&1 &

        MONITOR_PIDS+=("$!")
        echo "Started rdmatop monitor on $NODE (pid $!), logging to $LOGFILE"
    done

    # Give the monitors a moment to attach before the real traffic starts.
    sleep 3
}

stop_rdmatop_monitors() {
    for PID in "${MONITOR_PIDS[@]}"; do
        kill "$PID" 2>/dev/null
    done
    # The tmux server on each node keeps the session alive independent of
    # the srun wrapper above, so kill it explicitly too. We started it on
    # a job-scoped socket, so kill-server there tears down cleanly without
    # touching any other tmux sessions the user has on that node.
    if [[ -n "$TMUX_BIN" ]]; then
        for NODE in "${MONITOR_NODES[@]}"; do
            srun --overlap --nodes=1 --ntasks=1 --nodelist="$NODE" \
                 tmux -L "rdmatop_${SLURM_JOB_ID}" kill-server 2>/dev/null
        done
    fi
    wait 2>/dev/null
    echo "Stopped rdmatop monitors."
}

# Ensure monitors are cleaned up even if the job errors out or is cancelled.
trap stop_rdmatop_monitors EXIT

start_rdmatop_monitors

# --- Pairwise sendrecv tests ---
# Runs one pair at a time so each srun gets a clean 2-rank communicator.
# --nodelist inside srun selects a subset of the job's allocated nodes;
# all three must have been granted by the #SBATCH --nodelist above.
#
# Parameters chosen to match the msccl-test run:
#   -b 3K -e 3G -f 2    : same message-size sweep
#   -g 1                 : one GPU per rank
#   -c 0                 : no data verification (matches -c 0 in msccl run)
#   -n 80 -w 80          : 80 warmup + 80 timed iterations
#   -G 0                 : no CUDA graph capture (caused hangs in msccl run)
#   -z 0                 : no out-of-place buffer (in-place, matches msccl)

PAIRS=("ja2,ja3" "ja2,ja4" "ja3,ja4")

for PAIR in "${PAIRS[@]}"; do
    echo ""
    echo "========================================================"
    echo "  Pairwise sendrecv: $PAIR"
    echo "  $(date)"
    echo "========================================================"

    srun --mpi=pmix \
         --ntasks=2 \
         --ntasks-per-node=1 \
         --nodelist="$PAIR" \
         --export=ALL \
         $HOME/nccl-tests/build/sendrecv_perf \
         -b 3K -e 3G -f 2 -g 1 -c 0 -n 80 -w 80 -G 0 -z 0

    echo ""
done

echo "========================================================"
echo "  All pairwise tests complete: $(date)"
echo "========================================================"

stop_rdmatop_monitors
