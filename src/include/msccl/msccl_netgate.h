/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#ifndef MSCCL_NETGATE_H_
#define MSCCL_NETGATE_H_

// MSCCL net gates: the XML attributes netdepid/netdeps on a send step say
// "do not put this send on the wire until that send (another threadblock on
// this GPU) has been acknowledged by the remote NIC, in the same pass".
//
// Host side, the gate is resolved in three steps:
//   parse  (bid, XML step) -> (bid, transmission)    mscclNetGateInfo::specs
//   bind   bid -> its send connection, per comm       mscclNetGateBound
//   lower  transmission -> proxy step within a pass  mscclNetGateLowered
// Each send op then carries a pointer to its threadblock's mscclNetGateOp.
// The proxy progress thread compares that against the progress the
// predecessor's connection publishes (sendNetResources::gateOpCount/gateDone
// in transport/net.cc) before every isend.

#include <cstdint>
#include <map>
#include <mutex>
#include <vector>
#include "nccl.h"

struct ncclProxyConnection;
struct mscclAlgo;

// NCCL_MSCCL_NET_GATES: 0 ignores netdepid/netdeps, 1 enforces them (default),
// 2 enforces them and logs each gate that opens after holding a send back.
int64_t ncclParamMscclNetGates();
// NCCL_MSCCL_NET_GATE_WARN_MS: warn once per op when a gate has held a ready
// send back for this long. 0 disables the warning.
int64_t ncclParamMscclNetGateWarnMs();

// A gate in transmission space: transmission `tran` of threadblock `bid` waits
// for transmission `depTran` of threadblock `depBid`. Both are sends.
struct mscclNetGateSpec {
  int16_t bid;
  int16_t tran;
  int16_t depBid;
  int16_t depTran;
};

// One wait condition in proxy-step space. Immutable once built.
struct mscclNetGate {
  struct ncclProxyConnection* depConn; // predecessor's send connection (proxy side)
  uint32_t atStep;                     // where in its pass the gated send starts
  uint32_t depEnd;                     // where in its pass the predecessor send ends (exclusive)
  uint32_t depStepsPerLoop;            // predecessor's proxy steps per pass
  int16_t depBid;                      // for log messages
};

// What one send op needs. Immutable once built; the proxy only reads it.
struct mscclNetGateOp {
  uint32_t stepsPerLoop;               // this op's proxy steps per pass
  int16_t bid;
  uint16_t nGates;                     // 0: publishes its progress, never waits
  const struct mscclNetGate* gates;    // sorted by atStep
};

// Per (comm, maxAllowedCount). gates is filled completely before any
// ops[bid].gates points into it, and neither changes afterwards.
struct mscclNetGateLowered {
  std::vector<struct mscclNetGate> gates;
  std::vector<struct mscclNetGateOp> ops;   // by bid
  std::vector<uint8_t> involved;            // by bid: gated, or the predecessor of a gate
};

// Per comm, built once when the algorithm's connections are set up.
struct mscclNetGateBound {
  std::vector<struct mscclNetGateSpec> specs;          // gates whose two ends are network sends
  std::vector<struct ncclProxyConnection*> conn;       // by bid: send connection, NULL if not network
  std::map<uint32_t, struct mscclNetGateLowered*> lowered;
};

// Owned by the host mscclAlgo; never read on the device.
struct mscclNetGateInfo {
  std::vector<struct mscclNetGateSpec> specs;          // sorted by (bid, tran)
  std::map<ncclComm_t, struct mscclNetGateBound> bound;
  std::mutex mutex;                                    // setup can also run on a CUDA graph host-node thread
};

ncclResult_t mscclNetGateBind(struct mscclAlgo* algo, ncclComm_t comm);
ncclResult_t mscclNetGateLower(struct mscclAlgo* algo, ncclComm_t comm, uint32_t maxAllowedCount,
                               int chunkSteps, int sliceSteps, const struct mscclNetGateLowered** out);
void mscclNetGateFree(struct mscclAlgo* algo);

#endif
