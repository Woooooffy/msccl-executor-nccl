/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INT_NET_H_
#define NCCL_INT_NET_H_

#include "nccl.h"
#include "nccl_net.h"
#include "comm.h"
#include "checks.h"

typedef char ncclNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

ncclResult_t ncclNetPluginLoad(struct ncclComm* comm);
ncclResult_t ncclNetPluginUnload(struct ncclComm* comm);
ncclResult_t ncclNetInit(struct ncclComm* comm);
ncclResult_t ncclNetFinalize(struct ncclComm* comm);
int ncclNetVersion(struct ncclComm* comm);

// Test whether the current GPU support GPU Direct RDMA.
ncclResult_t ncclGpuGdrSupport(struct ncclComm* comm, int* gdrSupport);

extern ncclNet_t ncclNetIb;
extern ncclNet_t ncclNetSocket;

// MSCCL extension to the built-in IB transport: per-send metadata the ncclNet
// isend has no argument for. Kept out of ncclNet_t so external plugins and the
// plugin ABI are untouched; the proxy uses it only when ncclNet == &ncclNetIb.
struct ncclNetSendMeta {
  uint32_t flowId;   // XML mscclflowid of this send, 0 = none
  uint32_t rateMBps; // XML rate of this send in MB/s, 0 = unpaced
};
typedef struct {
  // Same contract as ncclNet_t::isend, plus meta, which is copied before return.
  ncclResult_t (*isendMeta)(void* sendComm, void* data, int size, int tag, void* mhandle,
                            const struct ncclNetSendMeta* meta, void** request);
  // Declare that every receive posted on this registration has `bytes` of dead
  // memory before its address, which the sender may overwrite with a flow header.
  ncclResult_t (*setMrHeadroom)(void* mhandle, int bytes);
  // Rate control: post whatever fragments of this comm's paced sends are due
  // now, never waiting. *pending is 1 while fragments remain for a later call.
  // A send's completion (test) still comes only after all of it is posted.
  ncclResult_t (*progressSend)(void* sendComm, int* pending);
} ncclNetMscclExt_t;
extern ncclNetMscclExt_t ncclNetIbMscclExt;

// NCCL_MSCCL_FLOW_ID (defined in transport/net.cc): 0 off, 1 flow ids and the
// per-fragment flow header on the wire, 2 as 1 plus per-send/receive logging.
int64_t ncclParamMscclFlowId();
// NCCL_MSCCL_RATE (defined in transport/net.cc): 0 off, 1 pace each send at its
// XML rate inside the IB plugin, 2 as 1 plus per-send and per-connection logs.
int64_t ncclParamMscclRate();

// Flow header carried at the start of every fragment's payload. This struct is
// the wire format: fields are big-endian. Fragment i of a send is written to
// (address + i*D - sizeof(hdr)) and fragments are posted from last to first, so
// each header but fragment 0's is overwritten by the data of the fragment before
// it; fragment 0's lands in the receive pad before the slot.
struct ncclNetFlowHdr {
  uint32_t magic;  // NCCL_NET_FLOW_HDR_MAGIC
  uint32_t flowId; // XML mscclflowid, 0 = none (switch falls back to ECMP)
};
#define NCCL_NET_FLOW_HDR_MAGIC 0x4D53464Cu // "MSFL"
// Dead bytes reserved before each Simple receive slot. A 128B multiple, so slot
// data stays 128B aligned, and at least one header.
#define NCCL_NET_FLOW_HDR_PAD 128
static_assert(sizeof(struct ncclNetFlowHdr) <= NCCL_NET_FLOW_HDR_PAD, "flow header must fit the receive pad");
static_assert(NCCL_NET_FLOW_HDR_PAD % 128 == 0, "receive pad must keep slots 128B aligned");

#endif
