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
  uint32_t flowId; // XML mscclflowid of this send, 0 = none
};
typedef struct {
  // Same contract as ncclNet_t::isend, plus meta, which is copied before return.
  ncclResult_t (*isendMeta)(void* sendComm, void* data, int size, int tag, void* mhandle,
                            const struct ncclNetSendMeta* meta, void** request);
} ncclNetMscclExt_t;
extern ncclNetMscclExt_t ncclNetIbMscclExt;

#endif
