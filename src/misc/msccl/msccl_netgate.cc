/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include "comm.h"
#include "param.h"
#include "proxy.h"
#include "transport.h"
#include "msccl/msccl_netgate.h"
#include "msccl/msccl_struct.h"

NCCL_PARAM(MscclNetGates, "MSCCL_NET_GATES", 1);
NCCL_PARAM(MscclNetGateWarnMs, "MSCCL_NET_GATE_WARN_MS", 10000);

static bool mscclNetGateIsSend(uint8_t type) {
  return type == MSCCL_SEND || type == MSCCL_RECV_COPY_SEND ||
         type == MSCCL_RECV_REDUCE_SEND || type == MSCCL_RECV_REDUCE_COPY_SEND;
}

static void mscclNetGateFreeLowered(struct mscclNetGateBound* bound) {
  for (auto& kv : bound->lowered) delete kv.second;
  bound->lowered.clear();
}

// Called once per (comm, algo), after the algorithm's connections are set up.
ncclResult_t mscclNetGateBind(struct mscclAlgo* algo, ncclComm_t comm) {
  struct mscclNetGateInfo* info = algo->netGates;
  if (info == NULL || ncclParamMscclNetGates() == 0) return ncclSuccess;
  std::lock_guard<std::mutex> lock(info->mutex);

  struct mscclNetGateBound& bound = info->bound[comm];
  mscclNetGateFreeLowered(&bound);
  bound.specs.clear();
  bound.conn.assign(algo->nBlocks, NULL);
  // PXN routes a send through another GPU's proxy, which in MSCCL (one GPU per
  // process) is another process: the gate could neither see the predecessor's
  // progress nor dereference the pointers it is handed.
  std::vector<uint8_t> pxn(algo->nBlocks, 0);
  for (int bid = 0; bid < algo->nBlocks; bid++) {
    struct mscclThreadBlock* tb = &algo->mscclTBs[bid];
    if (tb->sendPeer < 0) continue;
    struct ncclConnector* conn = comm->channels[tb->channelId].peers[tb->sendPeer]->send + 0;
    if (conn->transportComm != &netTransport.send) continue;
    bound.conn[bid] = conn->proxyConn.connection;
    pxn[bid] = conn->proxyConn.rank != comm->rank;
  }

  int droppedGated = 0, droppedPredecessor = 0;
  for (auto& g : info->specs) {
    // The proxy only sees network sends; an NVLink or SHM end cannot be gated on.
    if (bound.conn[g.bid] == NULL) { droppedGated++; continue; }
    if (bound.conn[g.depBid] == NULL) { droppedPredecessor++; continue; }
    if (pxn[g.bid] || pxn[g.depBid]) {
      WARN("MSCCL: net gate threadblock %d <- threadblock %d on rank %d: a send is routed through another GPU's proxy (PXN). "
           "Set NCCL_PXN_DISABLE=1, or NCCL_MSCCL_NET_GATES=0 to ignore net gates.", g.bid, g.depBid, comm->rank);
      return ncclInvalidUsage;
    }
    bound.specs.push_back(g);
  }
  INFO(NCCL_INIT, "MSCCL: rank %d net gates: %zu kept, %d dropped (gated send not on network %d, predecessor not on network %d)",
       comm->rank, bound.specs.size(), droppedGated + droppedPredecessor, droppedGated, droppedPredecessor);
  return ncclSuccess;
}

// Called by every mscclSetupProxyImpl. Builds the proxy-step form of the gates
// the first time a comm runs the algorithm with this maxAllowedCount; returns
// the cached, immutable result afterwards.
ncclResult_t mscclNetGateLower(struct mscclAlgo* algo, ncclComm_t comm, uint32_t maxAllowedCount,
                               int chunkSteps, int sliceSteps, const struct mscclNetGateLowered** out) {
  *out = NULL;
  struct mscclNetGateInfo* info = algo->netGates;
  if (info == NULL || ncclParamMscclNetGates() == 0) return ncclSuccess;
  std::lock_guard<std::mutex> lock(info->mutex);

  auto boundIt = info->bound.find(comm);
  if (boundIt == info->bound.end()) {
    WARN("MSCCL: net gates were not bound to this communicator (rank %d)", comm->rank);
    return ncclInternalError;
  }
  struct mscclNetGateBound& bound = boundIt->second;
  if (bound.specs.empty()) return ncclSuccess;
  auto loweredIt = bound.lowered.find(maxAllowedCount);
  if (loweredIt != bound.lowered.end()) {
    *out = loweredIt->second;
    return ncclSuccess;
  }

  // Proxy steps of one pass, per sending threadblock: a send transmission makes
  // DIVUP(count, maxAllowedCount) prim calls of chunkSteps steps each
  // (msccl_kernel.cu); anything else uses no steps on the send connection.
  int nBlocks = algo->nBlocks;
  std::vector<std::vector<uint32_t>> offset(nBlocks), span(nBlocks);
  std::vector<uint32_t> stepsPerLoop(nBlocks, 0);
  for (int bid = 0; bid < nBlocks; bid++) {
    struct mscclThreadBlock* tb = &algo->mscclTBs[bid];
    if (tb->sendPeer < 0) continue;
    offset[bid].resize(tb->nSteps);
    span[bid].resize(tb->nSteps);
    uint32_t total = 0;
    for (int i = 0; i < tb->nSteps; i++) {
      struct mscclTransmission* t = &tb->transmissions[i];
      span[bid][i] = mscclNetGateIsSend(t->type) ? chunkSteps * DIVUP((uint32_t)t->count, maxAllowedCount) : 0;
      offset[bid][i] = total;
      total += span[bid][i];
    }
    stepsPerLoop[bid] = total;
  }
  // Must match the nsteps mscclSetupProxyImpl gives each send op, or gates would
  // fire at the wrong steps without any error.
  for (int ch = 0; ch < algo->nChannels; ch++) {
    struct mscclChannelInfo* mCh = algo->mscclChannels + ch;
    for (int p = 0; p < mCh->nSendPeers; p++) {
      struct mscclChannelPeerInfo* peer = mCh->sendPeerInfo + p;
      uint32_t nSends = 0;
      for (int j = 0; j < peer->nExistingCounts; j++) {
        int c = peer->existingCounts[j];
        nSends += peer->nTransmissionsOfCount[c] * DIVUP((uint32_t)c, maxAllowedCount);
      }
      if (stepsPerLoop[peer->bid] != chunkSteps * nSends) {
        WARN("MSCCL: net gates: threadblock %d has %u proxy steps per pass, but its proxy op expects %u",
             peer->bid, stepsPerLoop[peer->bid], chunkSteps * nSends);
        return ncclInternalError;
      }
    }
  }

  struct mscclNetGateLowered* lowered = new mscclNetGateLowered();
  lowered->ops.assign(nBlocks, mscclNetGateOp{0, -1, 0, NULL});
  lowered->involved.assign(nBlocks, 0);
  lowered->gates.reserve(bound.specs.size());
  std::vector<size_t> first(nBlocks, 0);
  for (int bid = 0; bid < nBlocks; bid++) {
    lowered->ops[bid].stepsPerLoop = stepsPerLoop[bid];
    lowered->ops[bid].bid = bid;
  }
  for (auto& g : bound.specs) { // sorted by (bid, tran), so each bid's gates are contiguous and by atStep
    struct mscclNetGate gate;
    gate.depConn = bound.conn[g.depBid];
    gate.atStep = offset[g.bid][g.tran];
    gate.depEnd = offset[g.depBid][g.depTran] + span[g.depBid][g.depTran];
    gate.depStepsPerLoop = stepsPerLoop[g.depBid];
    gate.depBid = g.depBid;
    if (gate.atStep % sliceSteps != 0 || gate.depEnd == 0 || gate.depEnd > gate.depStepsPerLoop) {
      WARN("MSCCL: net gate threadblock %d <- %d lowered to atStep %u, depEnd %u of %u (sliceSteps %d)",
           g.bid, g.depBid, gate.atStep, gate.depEnd, gate.depStepsPerLoop, sliceSteps);
      delete lowered;
      return ncclInternalError;
    }
    if (lowered->ops[g.bid].nGates == 0) first[g.bid] = lowered->gates.size();
    lowered->ops[g.bid].nGates++;
    lowered->involved[g.bid] = 1;
    lowered->involved[g.depBid] = 1;
    lowered->gates.push_back(gate);
  }
  // gates is final; only now take pointers into it.
  for (int bid = 0; bid < nBlocks; bid++) {
    if (lowered->ops[bid].nGates) lowered->ops[bid].gates = &lowered->gates[first[bid]];
  }
  bound.lowered[maxAllowedCount] = lowered;
  *out = lowered;
  INFO(NCCL_INIT|NCCL_NET, "MSCCL: rank %d net gates lowered for maxAllowedCount %u: %zu gates", comm->rank, maxAllowedCount, lowered->gates.size());
  return ncclSuccess;
}

// Called before the host mscclAlgo is freed, after the proxies have stopped
// using any op that points into it.
void mscclNetGateFree(struct mscclAlgo* algo) {
  if (algo == NULL || algo->netGates == NULL) return;
  for (auto& kv : algo->netGates->bound) mscclNetGateFreeLowered(&kv.second);
  delete algo->netGates;
  algo->netGates = NULL;
}
