/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>
#include <algorithm>
#include <vector>
#include "core.h"
#include "collectives.h"
#include "msccl/msccl_parser.h"
#include "msccl/msccl_netgate.h"

ncclResult_t mscclXmlGetChar(FILE* file, char* c) {
  if (fread(c, 1, 1, file) == 0) {
    WARN("XML Parse : Unexpected EOF");
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t mscclXmlGetValue(FILE* file, char* value, char* last) {
  char c;
  NCCLCHECK(mscclXmlGetChar(file, &c));
  if (c != '"' && c != '\'') {
#if INT_OK
    int o = 0;
    do {
      value[o++] = c;
      NCCLCHECK(mscclXmlGetChar(file, &c));
    } while (c >= '0' && c <= '9');
    value[o] = '\0';
    *last = c;
    return ncclSuccess;
#else
    WARN("XML Parse : Expected (double) quote.");
    return ncclInternalError;
#endif
  }
  int o = 0;
  do {
    NCCLCHECK(mscclXmlGetChar(file, &c));
    value[o++] = c;
  } while (c != '"');
  value[o-1] = '\0';
  NCCLCHECK(mscclXmlGetChar(file, last));
  return ncclSuccess;
}

ncclResult_t mscclXmlGetToken(FILE* file, char* name, char* value, char* last) {
  char c;
  char* ptr = name;
  int o = 0;
  do {
    NCCLCHECK(mscclXmlGetChar(file, &c));
    if (c == '=') {
      ptr[o] = '\0';
      if (value == NULL) {
        WARN("XML Parse : Unexpected value with name %s", ptr);
        return ncclInternalError;
      }
      return mscclXmlGetValue(file, value, last);
    }
    ptr[o] = c;
    if (o == MAX_STR_LEN-1) {
      ptr[o] = '\0';
      WARN("Error : name %s too long (max %d)", ptr, MAX_STR_LEN);
      return ncclInternalError;
    }
    o++;
  } while (c != ' ' && c != '>' && c != '/' && c != '\n' && c != '\r');
  ptr[o-1] = '\0';
  *last = c;
  return ncclSuccess;
}

// Shift the 3-chars string by one char and append c at the end
#define SHIFT_APPEND(s, c) do { s[0]=s[1]; s[1]=s[2]; s[2]=c; } while(0)
ncclResult_t mscclXmlSkipComment(FILE* file, char* start, char next) {
  // Start from something neutral with \0 at the end.
  char end[4] = "...";

  // Inject all trailing chars from previous reads. We don't need
  // to check for --> here because there cannot be a > in the name.
  for (int i=0; i<strlen(start); i++) SHIFT_APPEND(end, start[i]);
  SHIFT_APPEND(end, next);

  // Stop when we find "-->"
  while (strcmp(end, "-->") != 0) {
    int c;
    if (fread(&c, 1, 1, file) != 1) {
      WARN("XML Parse error : unterminated comment");
      return ncclInternalError;
    }
    SHIFT_APPEND(end, c);
  }
  return ncclSuccess;
}

ncclResult_t mscclXmlGetNode(FILE* file, struct mscclXmlNode* node) {
  node->type = NODE_TYPE_NONE;
  char c = ' ';
  while (c == ' ' || c == '\n' || c == '\r') {
    if (fread(&c, 1, 1, file) == 0) return ncclSuccess;
  }
  if (c != '<') {
    WARN("XML Parse error : expecting '<', got '%c'", c);
    return ncclInternalError;
  }
  // Read XML element name
  NCCLCHECK(mscclXmlGetToken(file, node->name, NULL, &c));

  // Check for comments
  if (strncmp(node->name, "!--", 3) == 0) {
    NCCLCHECK(mscclXmlSkipComment(file, node->name+3, c));
    return mscclXmlGetNode(file, node);
  }

  // Check for closing tag
  if (node->name[0] == '\0' && c == '/') {
    node->type = NODE_TYPE_CLOSE;
    // Re-read the name, we got '/' in the first call
    NCCLCHECK(mscclXmlGetToken(file, node->name, NULL, &c));
    if (c != '>') {
      WARN("XML Parse error : unexpected trailing %c in closing tag %s", c, node->name);
      return ncclInternalError;
    }
    return ncclSuccess;
  }

  node->type = NODE_TYPE_OPEN;

  // Get Attributes
  int a = 0;
  while (c == ' ') {
    NCCLCHECK(mscclXmlGetToken(file, node->attrs[a].key, node->attrs[a].value, &c));
    if (a == MAX_ATTR_COUNT) {
      // Loud on purpose: a dropped attribute such as netdepid or mscclflowid
      // silently changes what the algorithm does.
      WARN("XML Parse : node %s has more than %d attributes; ignoring the extra ones", node->name, MAX_ATTR_COUNT);
      // Actually we need to still consume the extra attributes so we have an extra one.
    } else a++;
  }
  node->nAttrs = a;
  if (c == '/') {
    node->type = NODE_TYPE_SINGLE;
    char str[MAX_STR_LEN];
    NCCLCHECK(mscclXmlGetToken(file, str, NULL, &c));
  }
  if (c != '>') {
    WARN("XML Parse : expected >, got '%c'", c);
    return ncclInternalError;
  }
  return ncclSuccess;
}

typedef ncclResult_t (*mscclXmlHandlerFunc_t)(FILE*, struct mscclXml*, struct mscclXmlNode*);

struct mscclXmlHandler {
  const char * name;
  mscclXmlHandlerFunc_t func;
};

ncclResult_t mscclXmlLoadSub(FILE* file, struct mscclXml* xml, struct mscclXmlNode* head, struct mscclXmlHandler handlers[], int nHandlers) {
  if (head && head->type == NODE_TYPE_SINGLE) return ncclSuccess;
  while (1) {
    if (xml->maxIndex == MAX_NODES) {
      WARN("Error : XML parser is limited to 1024 nodes");
      return ncclInternalError;
    }
    struct mscclXmlNode* node = xml->nodes+xml->maxIndex;
    memset(node, 0, sizeof(struct mscclXmlNode));
    NCCLCHECK(mscclXmlGetNode(file, node));
    if (node->type == NODE_TYPE_NONE) {
      if (head) {
        WARN("XML Parse : unterminated %s", head->name);
        return ncclInternalError;
      } else {
        // All done
        return ncclSuccess;
      }
    }
    if (head && node->type == NODE_TYPE_CLOSE) {
      if (strcmp(node->name, head->name) != 0) {
        WARN("XML Mismatch : %s / %s", head->name, node->name);
        return ncclInternalError;
      }
      return ncclSuccess;
    }
    int found = 0;
    for (int h=0; h<nHandlers; h++) {
      if (strcmp(node->name, handlers[h].name) == 0) {
        if (head) head->subs[head->nSubs++] = node;
        node->parent = head;
        node->nSubs = 0;
        xml->maxIndex++;
        NCCLCHECK(handlers[h].func(file, xml, node));
        found = 1;
        break;
      }
    }
    if (!found) {
      if (nHandlers) INFO(NCCL_GRAPH, "Ignoring element %s", node->name);
      NCCLCHECK(mscclXmlLoadSub(file, xml, node, NULL, 0));
    }
  }
}

ncclResult_t mscclAlgoXmlStep(FILE* file, struct mscclXml* xml, struct mscclXmlNode* head) {
  NCCLCHECK(mscclXmlLoadSub(file, xml, head, NULL, 1));
  return ncclSuccess;
}

ncclResult_t mscclAlgoXmlThreadBlock(FILE* file, struct mscclXml* xmlGraph, struct mscclXmlNode* head) {
  struct mscclXmlHandler handlers[] = { { "step", mscclAlgoXmlStep } };
  NCCLCHECK(mscclXmlLoadSub(file, xmlGraph, head, handlers, 1));
  return ncclSuccess;
}

static int currentRank;

ncclResult_t mscclAlgoXmlGpu(FILE* file, struct mscclXml* xmlGraph, struct mscclXmlNode* head) {
  int thisrank;
  NCCLCHECK(mscclXmlGetAttrInt(head, "id", &thisrank));
  if (thisrank == currentRank) {
    struct mscclXmlHandler handlers[] = { { "tb", mscclAlgoXmlThreadBlock } };
    NCCLCHECK(mscclXmlLoadSub(file, xmlGraph, head, handlers, 1));
  } else {
    NCCLCHECK(mscclXmlLoadSub(file, xmlGraph, head, NULL, 0));
  }
  return ncclSuccess;
}

ncclResult_t mscclAlgoXmlAlgo(FILE* file, struct mscclXml* xmlGraph, struct mscclXmlNode* head) {
  struct mscclXmlHandler handlers[] = { { "gpu", mscclAlgoXmlGpu } };
  NCCLCHECK(mscclXmlLoadSub(file, xmlGraph, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t mscclAlgoXmlLoad(const char* xmlFilePath, struct mscclXml* xml, int rank) {
  currentRank = rank;
  FILE* file = fopen(xmlFilePath, "r");
  if (file == NULL) {
    WARN("Could not open MSCCL XML algorithm file %s : %s", xmlFilePath, strerror(errno));
    return ncclSystemError;
  }
  struct mscclXmlHandler handlers[] = { { "algo", mscclAlgoXmlAlgo } };
  xml->maxIndex = 0;
  NCCLCHECK(mscclXmlLoadSub(file, xml, NULL, handlers, 1));
  fclose(file);
  return ncclSuccess;
}

ncclResult_t mscclGetBufferType(const char* str, uint8_t* output) {
  if (strcmp(str, "i") == 0) {
    *output = MSCCL_INPUT_BUFFER;
  } else if (strcmp(str, "o") == 0) {
    *output = MSCCL_OUTPUT_BUFFER;
  } else if (strcmp(str, "s") == 0) {
    *output = MSCCL_SCRATCH_BUFFER;
  } else {
    WARN("type of buffer is not supported: %s", str);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t mscclCheckBufferBounds(int bufferType, int offset, int nInputChunks, int nOutputChunks, int nScratchChunks) {
  if (bufferType == MSCCL_INPUT_BUFFER) {
    if (offset < -1 || offset >= nInputChunks) {
      WARN("Incorrect offset set for input buffer: offset: %d maximum allowed: %d", offset, nInputChunks);
      return ncclInvalidUsage;
    }
  } else if (bufferType == MSCCL_OUTPUT_BUFFER) {
    if (offset < -1 || offset >= nOutputChunks) {
      WARN("Incorrect offset set for output buffer: offset: %d maximum allowed: %d", offset, nOutputChunks);
      return ncclInvalidUsage;
    }
  } else if (bufferType == MSCCL_SCRATCH_BUFFER) {
    if (offset < -1 || offset >= nScratchChunks) {
      WARN("Incorrect offset set for scratch buffer: offset: %d maximum allowed: %d", offset, nScratchChunks);
      return ncclInvalidUsage;
    }
  }
  return ncclSuccess;
}

ncclResult_t mscclProtocolStrToId(const char *protocol, int *protocolId) {
  if (strcmp(protocol, "Simple") == 0) {
    *protocolId = NCCL_PROTO_SIMPLE;
  } else if (strcmp(protocol, "LL128") == 0) {
    *protocolId = NCCL_PROTO_LL128;
  } else if (strcmp(protocol, "LL") == 0) {
    *protocolId = NCCL_PROTO_LL;
  } else {
    WARN("MSCCL: protocol %s is not supported.", protocol);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

// A netdepid/netdeps pair as read from the XML. Steps are document positions
// within the threadblock, which is also the kernel's step counter.
struct mscclNetGateRaw {
  int bid, pos, depBid, depPos;
};

static bool mscclIsSendType(uint8_t type) {
  return type == MSCCL_SEND || type == MSCCL_RECV_COPY_SEND ||
         type == MSCCL_RECV_REDUCE_SEND || type == MSCCL_RECV_REDUCE_COPY_SEND;
}

// Transmission at XML position `pos` of threadblock `bid` if it is a send that
// moves data (cnt > 0), else -1. A nop maps to -1.
static int mscclNetGateSendTran(struct mscclAlgo* algo, const std::vector<std::vector<int16_t>>& stepToTran, int bid, int pos) {
  if (bid < 0 || bid >= algo->nBlocks || pos < 0 || pos >= (int)stepToTran[bid].size()) return -1;
  int tran = stepToTran[bid][pos];
  if (tran < 0) return -1;
  struct mscclTransmission* t = &algo->mscclTBs[bid].transmissions[tran];
  return (mscclIsSendType(t->type) && t->count > 0) ? tran : -1;
}

// Transmission that completes the kernel's step counter value `pos` in `bid`:
// the one at that position, or the next real one after a nop. -1 if none.
static int mscclTranAtOrAfter(const std::vector<std::vector<int16_t>>& stepToTran, int bid, int pos) {
  if (pos < 0) return -1;
  for (int p = pos; p < (int)stepToTran[bid].size(); p++) {
    if (stepToTran[bid][p] >= 0) return stepToTran[bid][p];
  }
  return -1;
}

// A gate can deadlock the proxy; a data dependency can deadlock the kernel. Reject a
// cycle over (threadblock program order) + (depid/deps) + (net gates) on this GPU.
// Every edge is within one pass or points forward, so one pass is enough.
// Cycles that cross GPUs through the receivers' buffers are not visible here.
static ncclResult_t mscclCheckNetGatesAcyclic(struct mscclAlgo* algo, int rank,
    const std::vector<std::vector<int16_t>>& stepToTran, const std::vector<struct mscclNetGateSpec>& specs) {
  int nBlocks = algo->nBlocks;
  std::vector<int> base(nBlocks + 1, 0);
  for (int b = 0; b < nBlocks; b++) base[b+1] = base[b] + algo->mscclTBs[b].nSteps;
  int nNodes = base[nBlocks];
  std::vector<std::vector<int>> succ(nNodes);
  for (int b = 0; b < nBlocks; b++) {
    struct mscclThreadBlock* tb = &algo->mscclTBs[b];
    for (int i = 0; i < tb->nSteps; i++) {
      if (i + 1 < tb->nSteps) succ[base[b]+i].push_back(base[b]+i+1);
      struct mscclTransmission* t = &tb->transmissions[i];
      for (int d = t->dependencePointer; d < t->dependencePointer + t->numDependencies; d++) {
        int dBid = tb->dependentBid[d];
        if (dBid < 0 || dBid >= nBlocks) continue;
        int dTran = mscclTranAtOrAfter(stepToTran, dBid, tb->dependentStep[d]);
        if (dTran >= 0) succ[base[dBid]+dTran].push_back(base[b]+i);
      }
    }
  }
  for (auto& g : specs) succ[base[g.depBid]+g.depTran].push_back(base[g.bid]+g.tran);

  // Iterative DFS; on a back edge, print the cycle.
  std::vector<uint8_t> color(nNodes, 0); // 0 unvisited, 1 on stack, 2 done
  std::vector<std::pair<int, size_t>> stack;
  for (int root = 0; root < nNodes; root++) {
    if (color[root]) continue;
    color[root] = 1;
    stack.push_back({root, 0});
    while (!stack.empty()) {
      int node = stack.back().first;
      size_t& next = stack.back().second;
      if (next == succ[node].size()) {
        color[node] = 2;
        stack.pop_back();
        continue;
      }
      int child = succ[node][next++];
      if (color[child] == 1) {
        char msg[1024];
        int len = 0;
        size_t start = 0;
        while (stack[start].first != child) start++;
        for (size_t s = start; s < stack.size() && len < (int)sizeof(msg) - 32; s++) {
          int n = stack[s].first;
          int b = std::upper_bound(base.begin(), base.end(), n) - base.begin() - 1;
          len += snprintf(msg + len, sizeof(msg) - len, "TB %d tran %d -> ", b, n - base[b]);
        }
        snprintf(msg + len, sizeof(msg) - len, "...");
        WARN("MSCCL: net gates on GPU %d would deadlock; cycle over program order, depid/deps and netdepid/netdeps: %s", rank, msg);
        return ncclInvalidUsage;
      }
      if (color[child] == 0) {
        color[child] = 1;
        stack.push_back({child, 0});
      }
    }
  }
  return ncclSuccess;
}

// Turn the raw XML gates into transmission-space specs on algo->netGates.
static ncclResult_t mscclResolveNetGates(struct mscclAlgo* algo, int rank, const std::vector<struct mscclNetGateRaw>& raw,
    const std::vector<std::vector<int16_t>>& stepToTran, int sMismatchBid, int sMismatchPos, int sMismatchS) {
  if (raw.empty()) return ncclSuccess;
  // netdeps is the producer's step counter; resolving it by position is only
  // right if the XML numbers its steps by position, as ncclize does.
  if (sMismatchBid >= 0) {
    WARN("MSCCL: GPU %d threadblock %d has s=%d at step position %d; netdepid/netdeps need s to equal the step position",
         rank, sMismatchBid, sMismatchS, sMismatchPos);
    return ncclInvalidUsage;
  }
  // Two sending threadblocks on one (channel, peer) share a connection, and the
  // second one's proxy op runs only after the first one's retires.
  for (int b1 = 0; b1 < algo->nBlocks; b1++) {
    struct mscclThreadBlock* t1 = &algo->mscclTBs[b1];
    if (t1->sendPeer < 0) continue;
    for (int b2 = b1 + 1; b2 < algo->nBlocks; b2++) {
      struct mscclThreadBlock* t2 = &algo->mscclTBs[b2];
      if (t2->sendPeer == t1->sendPeer && t2->channelId == t1->channelId) {
        WARN("MSCCL: GPU %d threadblocks %d and %d both send to %d on channel %d; net gates need one sending threadblock per connection",
             rank, b1, b2, t1->sendPeer, t1->channelId);
        return ncclInvalidUsage;
      }
    }
  }

  std::vector<struct mscclNetGateSpec> specs;
  for (auto& r : raw) {
    int tran = mscclNetGateSendTran(algo, stepToTran, r.bid, r.pos);
    if (tran < 0) {
      WARN("MSCCL: GPU %d threadblock %d step %d has netdepid, but only a send with cnt > 0 can be gated", rank, r.bid, r.pos);
      return ncclInvalidUsage;
    }
    if (r.depBid < 0 || r.depBid >= algo->nBlocks || algo->mscclTBs[r.depBid].sendPeer < 0) {
      WARN("MSCCL: GPU %d threadblock %d step %d: netdepid=%d is not a sending threadblock", rank, r.bid, r.pos, r.depBid);
      return ncclInvalidUsage;
    }
    int depTran = mscclNetGateSendTran(algo, stepToTran, r.depBid, r.depPos);
    if (depTran < 0) {
      WARN("MSCCL: GPU %d threadblock %d step %d: netdeps=%d names threadblock %d step %d, which is not a send with cnt > 0",
           rank, r.bid, r.pos, r.depPos, r.depBid, r.depPos);
      return ncclInvalidUsage;
    }
    if (r.depBid == r.bid) {
      INFO(NCCL_INIT, "MSCCL: GPU %d threadblock %d step %d: netdep on its own threadblock dropped; one connection is already in order",
           rank, r.bid, r.pos);
      continue;
    }
    specs.push_back({(int16_t)r.bid, (int16_t)tran, (int16_t)r.depBid, (int16_t)depTran});
  }
  // By (bid, tran): a threadblock's gates are contiguous and, since proxy-step
  // offsets grow with the transmission index, sorted by atStep once lowered.
  std::sort(specs.begin(), specs.end(), [](const struct mscclNetGateSpec& a, const struct mscclNetGateSpec& b) {
    return a.bid != b.bid ? a.bid < b.bid : a.tran < b.tran;
  });
  NCCLCHECK(mscclCheckNetGatesAcyclic(algo, rank, stepToTran, specs));
  if (specs.empty()) return ncclSuccess;

  algo->netGates = new mscclNetGateInfo();
  algo->netGates->specs = std::move(specs);
  INFO(NCCL_INIT, "MSCCL: GPU %d: %zu net gates parsed", rank, algo->netGates->specs.size());
  return ncclSuccess;
}

ncclResult_t mscclGetAlgoFromXmlFile(const char* str, struct mscclAlgo* algo, int rank) {
  struct mscclXml* xml;
  NCCLCHECK(ncclCalloc(&xml, 1));
  NCCLCHECK(mscclAlgoXmlLoad(str, xml, rank));

  // zeroing out all entries.
  memset(algo, 0, sizeof(struct mscclAlgo));
  struct mscclXmlNode* topNode;
  NCCLCHECK(mscclXmlFindTag(xml, "algo", &topNode));

  int nChunksPerLoop;
  NCCLCHECK(mscclXmlGetAttrInt(topNode, "nchunksperloop", &nChunksPerLoop));
  algo->nChunksPerLoop  = nChunksPerLoop;

  int nChannels;
  NCCLCHECK(mscclXmlGetAttrInt(topNode, "nchannels", &nChannels));
  algo->nChannels = nChannels;

  int nGpus;
  NCCLCHECK(mscclXmlGetAttrInt(topNode, "ngpus", &nGpus));
  algo->nRanks = nGpus;

  const char* protocol;
  NCCLCHECK(mscclXmlGetAttrStr(topNode, "proto", &protocol));
  NCCLCHECK(mscclProtocolStrToId(protocol, &algo->protocol));

  algo->sizeMultiplier = 1;
  algo->chunkSteps = MSCCL_CHUNKSTEPS;
  algo->sliceSteps = MSCCL_SLICESTEPS;
  const char* coll;
  NCCLCHECK(mscclXmlGetAttrStr(topNode, "coll", &coll));
  if (strcmp(coll, "reduce") == 0) {
    algo->chunkSteps = REDUCE_CHUNKSTEPS;
    algo->sliceSteps = REDUCE_SLICESTEPS;
    algo->func = mscclFuncReduce;
  } else if (strcmp(coll, "broadcast") == 0) {
    algo->chunkSteps = BROADCAST_CHUNKSTEPS;
    algo->sliceSteps = BROADCAST_SLICESTEPS;
    algo->func = mscclFuncBroadcast;
  } else if (strcmp(coll, "allreduce") == 0) {
    algo->chunkSteps = ALLREDUCE_CHUNKSTEPS;
    algo->sliceSteps = ALLREDUCE_SLICESTEPS;
    algo->func = mscclFuncAllReduce;
  } else if (strcmp(coll, "reducescatter") == 0) {
    algo->sizeMultiplier = nGpus;
    algo->chunkSteps = REDUCESCATTER_CHUNKSTEPS;
    algo->sliceSteps = REDUCESCATTER_SLICESTEPS;
    algo->func = mscclFuncReduceScatter;
  } else if (strcmp(coll, "allgather") == 0) {
    algo->sizeMultiplier = nGpus;
    algo->chunkSteps = ALLGATHER_CHUNKSTEPS;
    algo->sliceSteps = ALLGATHER_SLICESTEPS;
    algo->func = mscclFuncAllGather;
  } else if (strcmp(coll, "send") == 0) {
    algo->func = mscclFuncSend;
  } else if (strcmp(coll, "recv") == 0) {
    algo->func = mscclFuncRecv;
  } else if (strcmp(coll, "gather") == 0) {
    algo->func = mscclFuncGather;
  } else if (strcmp(coll, "scatter") == 0) {
    algo->func = mscclFuncScatter;
  } else if (strcmp(coll, "alltoall") == 0) {
    algo->sizeMultiplier = nGpus;
    algo->func = mscclFuncAllToAll;
  } else if (strcmp(coll, "alltoallv") == 0) {
    algo->func = mscclFuncAllToAllv;
  } else {
    WARN("MSCCL: unsupported collective: %s", coll);
    return ncclInvalidUsage;
  }

  int64_t minBytes;
  NCCLCHECK(mscclXmlGetAttrInt64(topNode, "minBytes", &minBytes));
  algo->minBytes = minBytes;

  int64_t maxBytes;
  NCCLCHECK(mscclXmlGetAttrInt64(topNode, "maxBytes", &maxBytes));
  algo->maxBytes = maxBytes;

  int inplace;
  NCCLCHECK(mscclXmlGetAttrInt(topNode, "inplace", &inplace));
  algo->inPlace = (bool)inplace;

  int outofplace;
  NCCLCHECK(mscclXmlGetAttrInt(topNode, "outofplace", &outofplace));
  algo->outOfPlace = (bool)outofplace;

  algo->hasReduce = false;

  for (int s=0; s<topNode->nSubs; s++) {
    struct mscclXmlNode* node = topNode->subs[s];
    if (strcmp(node->name, "gpu") == 0) {
      int blockExists[MSCCL_MAX_NUM_THREAD_BLOCKS];
      memset(blockExists, 0, sizeof(int[MSCCL_MAX_NUM_THREAD_BLOCKS]));
      int id, nScratchChunks, nInputChunks, nOutputChunks;
      NCCLCHECK(mscclXmlGetAttrInt(node, "id", &id));
      if (id == rank) {
        // Net gate scratch, freed when this function returns.
        bool readNetGates = ncclParamMscclNetGates() != 0;
        std::vector<struct mscclNetGateRaw> netGateRaw;
        std::vector<std::vector<int16_t>> stepToTran(MSCCL_MAX_NUM_THREAD_BLOCKS); // [bid][position] -> transmission, -1 for nop
        int sMismatchBid = -1, sMismatchPos = -1, sMismatchS = -1;
        NCCLCHECK(mscclXmlGetAttrInt(node, "i_chunks", &nInputChunks));
        NCCLCHECK(mscclXmlGetAttrInt(node, "o_chunks", &nOutputChunks));
        NCCLCHECK(mscclXmlGetAttrInt(node, "s_chunks", &nScratchChunks));
        if (nScratchChunks < 0) {
          WARN("MSCCL: nScratchChunks must be not negative. nScratchChunks: %d", nScratchChunks);
          return ncclInvalidUsage;
        }
        algo->nScratchChunks = nScratchChunks;
        for (int t=0; t<node->nSubs; t++) {
          struct mscclXmlNode* threadBlockNode = node->subs[t];
          if (strcmp(threadBlockNode->name, "tb") == 0) {
            int bid, recvPeer, sendPeer, channelId;
            NCCLCHECK(mscclXmlGetAttrInt(threadBlockNode, "id", &bid));
            NCCLCHECK(mscclXmlGetAttrInt(threadBlockNode, "recv", &recvPeer));
            NCCLCHECK(mscclXmlGetAttrInt(threadBlockNode, "send", &sendPeer));
            NCCLCHECK(mscclXmlGetAttrInt(threadBlockNode, "chan", &channelId));
            if (bid < 0) {
              WARN("MSCCL: bid must be not negative. bid: %d", bid);
              return ncclInvalidUsage;
            }
            if (bid >= MSCCL_MAX_NUM_THREAD_BLOCKS) {
              WARN("MSCCL: too many thread blocks are requested. Max thread blocks: %d", MSCCL_MAX_NUM_THREAD_BLOCKS);
              return ncclInvalidUsage;
            }
            if (blockExists[bid]) {
              WARN("MSCCL: duplicate thread block id %d for MSCCL", bid);
              return ncclInvalidUsage;
            }
            blockExists[bid] = 1;

            if (recvPeer == id || sendPeer == id) {
              WARN("MSCCL: peer (%d,%d) and gpu id (%d) must be different", recvPeer, sendPeer, id);
              return ncclInvalidUsage;
            }
            struct mscclThreadBlock* sTB = &algo->mscclTBs[bid];
            sTB->nSteps = 0;
            if (recvPeer < -1 || sendPeer < -1) {
              WARN("MSCCL: wrong recvPeer (%d) or sendPeer (%d) in thread block %d on gpu %d", recvPeer, sendPeer, bid, id);
              return ncclInvalidUsage;
            }

            if (recvPeer == id || sendPeer == id) {
              WARN("MSCCL: recvPeer (%d) or sendPeer (%d) for thread block %d cannot be gpu %d", recvPeer, sendPeer, bid, id);
              return ncclInvalidUsage;
            }

            sTB->recvPeer = recvPeer;
            sTB->sendPeer = sendPeer;
            if (channelId < 0 || channelId > MAXCHANNELS) {
              WARN("MSCCL: threadblock %d on GPU %d has an invalid channel %d", bid, id, channelId);
              return ncclInvalidUsage;
            }
            sTB->channelId = channelId;

            // setting the summary of the msccl algorithm in msccl channels
            mscclChannelInfo* mscclChannel = &algo->mscclChannels[sTB->channelId];

            int numDependencies = 0;
            int oldDependencePointer = 0; // Indicator of where the dependencies started for nop

            int oldReductionDstBuffer = -1; // Indicator of last reduction buffer name; -1 means that last one wasn't a compatible reduction
            int oldReductionDstOffset = -1; // Indicator of last reduction buffer index
            int oldReductionSrcBuffer = -1; //
            int numReductions = 0;

            int numTransfers = 0;
            uint32_t tbSendFlowId = 0; // first send flow id seen on this thread block
            bool tbFlowIdWarned = false;
            int stepPos = 0; // position among this threadblock's <step> nodes; the kernel's step counter
            for (int st=0; st<threadBlockNode->nSubs; st++) {
              struct mscclXmlNode* stepNode = threadBlockNode->subs[st];
              if (strcmp(stepNode->name, "step") == 0) {
                int pos = stepPos++;
                stepToTran[bid].push_back(-1);
                int s, srcOffset, dstOffset, dependBid, dependStep, hasDependence, count;
                const char* srcBuffer, * dstBuffer, * type;
                NCCLCHECK(mscclXmlGetAttrInt(stepNode, "s", &s));

                NCCLCHECK(mscclXmlGetAttrInt(stepNode, "srcoff", &srcOffset));
                NCCLCHECK(mscclXmlGetAttrStr(stepNode, "srcbuf", &srcBuffer));
                NCCLCHECK(mscclXmlGetAttrInt(stepNode, "dstoff", &dstOffset));
                NCCLCHECK(mscclXmlGetAttrStr(stepNode, "dstbuf", &dstBuffer));

                NCCLCHECK(mscclXmlGetAttrInt(stepNode, "cnt", &count));
                NCCLCHECK(mscclXmlGetAttrStr(stepNode, "type", &type));
                NCCLCHECK(mscclXmlGetAttrInt(stepNode, "depid", &dependBid));
                NCCLCHECK(mscclXmlGetAttrInt(stepNode, "deps", &dependStep));
                NCCLCHECK(mscclXmlGetAttrInt(stepNode, "hasdep", &hasDependence));
                uint32_t flowId;
                NCCLCHECK(mscclXmlGetFlowId(stepNode, &flowId));
                if (readNetGates) {
                  int netDepBid = -1, netDepStep = -1;
                  bool hasNetDepBid, hasNetDepStep;
                  NCCLCHECK(mscclXmlGetAttrIntOpt(stepNode, "netdepid", &netDepBid, &hasNetDepBid));
                  NCCLCHECK(mscclXmlGetAttrIntOpt(stepNode, "netdeps", &netDepStep, &hasNetDepStep));
                  if (hasNetDepBid != hasNetDepStep) {
                    WARN("MSCCL: GPU %d threadblock %d step %d has only one of netdepid/netdeps", id, bid, s);
                    return ncclInvalidUsage;
                  }
                  if (netDepBid >= 0) {
                    if (netDepStep < 0) {
                      WARN("MSCCL: GPU %d threadblock %d step %d: netdepid=%d needs netdeps >= 0, got %d", id, bid, s, netDepBid, netDepStep);
                      return ncclInvalidUsage;
                    }
                    netGateRaw.push_back({bid, pos, netDepBid, netDepStep});
                  }
                  if (s != pos && sMismatchBid < 0) {
                    sMismatchBid = bid; sMismatchPos = pos; sMismatchS = s;
                  }
                }

                if (s >= MSCCL_MAX_NUM_STEPS){
                  WARN("MSCCL: too many steps are requested. Max number of steps: %d, requested: %d", MSCCL_MAX_NUM_STEPS, s+1);
                  return ncclInternalError;
                }
                if (s < 0){
                  WARN("MSCCL: step must be positive: step %d", s);
                  return ncclInternalError;
                }

                int hasSend = 0;
                int hasRecv = 0;
                int checkSrc = 0;
                int checkDst = 0;
                int transferType = -1; // -1 indicate a nop
                if (strcmp(type, "s") == 0) {
                  transferType = MSCCL_SEND;
                  hasSend = 1;
                  checkSrc = 1;
                } else if (strcmp(type, "r") == 0) {
                  transferType = MSCCL_RECV;
                  hasRecv = 1;
                  checkDst = 1;
                } else if (strcmp(type, "rcs") == 0) {
                  transferType = MSCCL_RECV_COPY_SEND;
                  hasSend = 1;
                  hasRecv = 1;
                  checkDst = 1;
                } else if (strcmp(type, "rrs") == 0) {
                  transferType = MSCCL_RECV_REDUCE_SEND;
                  hasSend = 1;
                  hasRecv = 1;
                  checkSrc = 1;
                  algo->hasReduce = true;
                } else if (strcmp(type, "rrc") == 0) {
                  transferType = MSCCL_RECV_REDUCE_COPY;
                  hasRecv = 1;
                  algo->hasReduce = true;
                } else if (strcmp(type, "rrcs") == 0) {
                  transferType = MSCCL_RECV_REDUCE_COPY_SEND;
                  hasRecv = 1;
                  hasSend = 1;
                  checkSrc = 1;
                  checkDst = 1;
                  algo->hasReduce = true;
                } else if (strcmp(type, "cpy") == 0) {
                  transferType = MSCCL_LOCAL_COPY;
                  checkSrc = 1;
                  checkDst = 1;
                } else if (strcmp(type, "re") == 0) {
                  transferType = MSCCL_REDUCE;
                  checkSrc = 1;
                  checkDst = 1;
                  algo->hasReduce = true;
                } else if (strcmp(type, "nop") == 0) {
                  transferType = -1;
                } else {
                  WARN("MSCCL: type of transfer is not supported: %s", type);
                  return ncclInternalError;
                }

                // A flow id names a network flow, so it only makes sense on a step
                // that moves data to or from a peer.
                if (flowId != 0 && !hasSend && !hasRecv) {
                  WARN("MSCCL: step %d of thread block %d on GPU %d is of type %s and cannot carry mscclflowid %u",
                       s, bid, id, type, flowId);
                  return ncclInvalidUsage;
                }

                if (dependBid >= 0) {
                  sTB->dependentBid[numDependencies] = dependBid;
                  sTB->dependentStep[numDependencies] = dependStep;
                  numDependencies++;
                }

                uint8_t srcBufferInt = 0;
                uint8_t dstBufferInt = 0;
                NCCLCHECK(mscclGetBufferType(srcBuffer, &srcBufferInt));
                NCCLCHECK(mscclGetBufferType(dstBuffer, &dstBufferInt));

                int continuationOfReductions = 0;
                // Analyze to see if this is in the same list of reductions for them to be chained
                if (transferType == MSCCL_REDUCE) {
                  if (oldReductionDstBuffer == dstBufferInt && oldReductionDstOffset == dstOffset && oldReductionSrcBuffer == srcBufferInt && dependBid == -1) {
                    numTransfers--; // reuse the same transfer
                    continuationOfReductions = 1;
                  } else {
                    oldReductionDstBuffer = -1;
                    oldReductionDstOffset = -1;
                  }
                }

                if (transferType != -1) {
                  struct mscclTransmission* mscclTran = &sTB->transmissions[numTransfers];
                  mscclTran->type = transferType;
                  mscclTran->srcOffset = srcOffset;
                  mscclTran->srcBuffer = srcBufferInt;
                  mscclTran->srcOffset = srcOffset;
                  mscclTran->dstBuffer = dstBufferInt;
                  mscclTran->dstOffset = dstOffset;

                  if (count < 0 || count >= MSCCL_MAX_COUNT){
                    WARN("MSCCL: count (%d) must be positive and less than %d", count, MSCCL_MAX_COUNT);
                    return ncclInternalError;
                  }

                  mscclTran->count = count;
                  // The id is carried on the wire per send. On a recv-only step it
                  // names the incoming flow, which the runtime does not use.
                  sTB->sendFlowId[numTransfers] = hasSend ? flowId : 0;

                  // A thread block sends on one (channel, peer) connection, i.e.
                  // one QP. Two ids on it mean two switch paths under one RC QP,
                  // whose packets can then arrive out of order.
                  if (hasSend && flowId != 0) {
                    if (tbSendFlowId == 0) {
                      tbSendFlowId = flowId;
                    } else if (flowId != tbSendFlowId && !tbFlowIdWarned) {
                      WARN("MSCCL: thread block %d on GPU %d (channel %d, sendPeer %d) sends with more than one "
                           "mscclflowid (%u and %u); they share one connection, so their packets may be reordered",
                           bid, id, channelId, sendPeer, tbSendFlowId, flowId);
                      tbFlowIdWarned = true;
                    }
                  }

                  if (hasSend) {
                    if (sendPeer < 0) {
                      WARN("MSCCL: there is a send in thread block %d on GPU %d without a sendPeer.", bid, id);
                      return ncclInvalidUsage;
                    }
                    if (mscclChannel->nSendPeers >= MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL) {
                      WARN("MSCCL: too many sends per channel. Max allowed %d", MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL);
                      return ncclInvalidUsage;
                    }

                    struct mscclChannelPeerInfo* sendPeerInfo = &mscclChannel->sendPeerInfo[mscclChannel->nSendPeers];
                    sendPeerInfo->nTransmissionsOfCount[count]++;
                  }
                  if (hasRecv) {
                    if (recvPeer < 0) {
                      WARN("MSCCL: there is a recv in thread block %d on GPU %d without a recvPeer.", bid, id);
                      return ncclInvalidUsage;
                    }
                    if (mscclChannel->nRecvPeers >= MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL) {
                      WARN("MSCCL: too many recvs per channel. Max allowed %d", MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL);
                      return ncclInvalidUsage;
                    }
                    struct mscclChannelPeerInfo* recvPeerInfo = &mscclChannel->recvPeerInfo[mscclChannel->nRecvPeers];
                    recvPeerInfo->nTransmissionsOfCount[count]++;
                  }

                  if (checkSrc) NCCLCHECK(mscclCheckBufferBounds(mscclTran->srcBuffer, mscclTran->srcOffset, nInputChunks, nOutputChunks, nScratchChunks));
                  if (checkDst) NCCLCHECK(mscclCheckBufferBounds(mscclTran->dstBuffer, mscclTran->dstOffset, nInputChunks, nOutputChunks, nScratchChunks));

                  if (!continuationOfReductions) {
                    mscclTran->dependencePointer = oldDependencePointer;
                    mscclTran->numDependencies = numDependencies - oldDependencePointer;
                    if (mscclTran->numDependencies > 0 && dependBid < 0) {
                      WARN("MSCCL: when there is a chain of dependencies, the last reduction must be a part of the first immediate instruction. Detected for GPU %d, thread block %d, and step %d. XML will be ignored.", id, bid, s);
                      return ncclInvalidUsage;
                    }
                    oldDependencePointer = numDependencies;
                  }

                  // reduction related pointers
                  if (transferType != MSCCL_REDUCE) {
                    oldReductionDstBuffer = -1;
                    oldReductionDstOffset = -1;
                    oldReductionSrcBuffer = -1;
                  } else {
                    if (oldReductionDstBuffer == -1) { // if this is the first reduction
                      mscclTran->reductionPointer = numReductions;
                    }
                    sTB->reductionSrcOffsets[numReductions] = mscclTran->srcOffset;
                    numReductions++;
                    mscclTran->numReductions = numReductions - mscclTran->reductionPointer;

                    if (hasDependence || numReductions == MSCCL_MAX_REDUCE_FUSION) {
                      oldReductionDstBuffer = -1;
                      oldReductionDstOffset = -1;
                    } else {
                      oldReductionDstBuffer = mscclTran->dstBuffer;
                      oldReductionDstOffset = mscclTran->dstOffset;
                      oldReductionSrcBuffer = mscclTran->srcBuffer;
                    }
                  }


                  if (hasDependence != 0 && hasDependence != 1) {
                    WARN("MSCCL: hasDependence needs to be 0 or 1, but it was %d", hasDependence);
                    return ncclInternalError;
                  }
                  mscclTran->hasDependence = hasDependence;

                  // A fused reduction continuation reuses numTransfers, so it maps
                  // to the surviving transmission.
                  stepToTran[bid][pos] = numTransfers;
                  numTransfers++;
                  sTB->nSteps = numTransfers;
                }
              }
            }

            // finish up mscclChannel calculation

            for (int c = 1; c <= MSCCL_MAX_COUNT; c++) {
              struct mscclChannelPeerInfo* sendPeer = &mscclChannel->sendPeerInfo[mscclChannel->nSendPeers];
              if (sendPeer->nTransmissionsOfCount[c] > 0) {
                sendPeer->existingCounts[sendPeer->nExistingCounts] = c;
                sendPeer->nExistingCounts++;
              }
              struct mscclChannelPeerInfo* recvPeer = &mscclChannel->recvPeerInfo[mscclChannel->nRecvPeers];
              if (recvPeer->nTransmissionsOfCount[c] > 0) {
                recvPeer->existingCounts[recvPeer->nExistingCounts] = c;
                recvPeer->nExistingCounts++;
              }
            }

            if (sTB->sendPeer >= 0) {
              mscclChannel->sendPeerInfo[mscclChannel->nSendPeers].peer = sTB->sendPeer;
              mscclChannel->sendPeerInfo[mscclChannel->nSendPeers].bid = bid;
              mscclChannel->nSendPeers++;
            }
            if (sTB->recvPeer >= 0) {
              mscclChannel->recvPeerInfo[mscclChannel->nRecvPeers].peer = sTB->recvPeer;
              mscclChannel->recvPeerInfo[mscclChannel->nRecvPeers].bid = bid;
              mscclChannel->nRecvPeers++;
            }
          }
        }
        // make sure that thread blocks are in order. Something like 0, 2, 3 is not allowed.
        if (blockExists[0] == 1) {
          algo->nBlocks = 1;
        }
        for (int i = 1; i < MSCCL_MAX_NUM_THREAD_BLOCKS; i++) {
          if (blockExists[i] == 1 && blockExists[i-1] == 0) {
            WARN("MSCCL: thread block %d is missing", i);
            return ncclInvalidUsage;
          }
          if (blockExists[i] == 1) {
            algo->nBlocks = i+1;
          }
        }

        NCCLCHECK(mscclResolveNetGates(algo, id, netGateRaw, stepToTran, sMismatchBid, sMismatchPos, sMismatchS));
      }
    }
  }
  free(xml);
  return ncclSuccess;
}

ncclResult_t mscclXmlLoadSingleNode(FILE* file, struct mscclXmlNode* node) {
  memset(node, 0, sizeof(struct mscclXmlNode));
  return mscclXmlGetNode(file, node);
}

ncclResult_t mscclAlgoMetaXmlLoad(const char* xmlFilePath, struct mscclXmlNode* node) {
  FILE* file = fopen(xmlFilePath, "r");
  if (file == NULL) {
    fprintf(stderr, "Could not open MSCCL XML algorithm file %s : %s", xmlFilePath, strerror(errno));
    return ncclSystemError;
  }
  NCCLCHECK(mscclXmlLoadSingleNode(file, node));
  fclose(file);
  return ncclSuccess;
}

ncclResult_t mscclGetAlgoMetaFromXmlFile(const char* str, struct mscclAlgoMeta* algoMeta) {
  struct mscclXmlNode* node;
  node = (struct mscclXmlNode *)malloc(sizeof(struct mscclXmlNode));
  NCCLCHECK(mscclAlgoMetaXmlLoad(str, node));

  algoMeta->filePath = str;

  int nChunksPerLoop;
  NCCLCHECK(mscclXmlGetAttrInt(node, "nchunksperloop", &nChunksPerLoop));
  algoMeta->nChunksPerLoop  = nChunksPerLoop;

  int nChannels;
  NCCLCHECK(mscclXmlGetAttrInt(node, "nchannels", &nChannels));
  algoMeta->nChannels = nChannels;

  int nGpus;
  NCCLCHECK(mscclXmlGetAttrInt(node, "ngpus", &nGpus));
  algoMeta->nRanks = nGpus;

  const char* coll;
  NCCLCHECK(mscclXmlGetAttrStr(node, "coll", &coll));
  algoMeta->sizeMultiplier = 1;
  if (strcmp(coll, "reduce") == 0) {
    algoMeta->func = mscclFuncReduce;
  } else if (strcmp(coll, "broadcast") == 0) {
    algoMeta->func = mscclFuncBroadcast;
  } else if (strcmp(coll, "allreduce") == 0) {
    algoMeta->func = mscclFuncAllReduce;
  } else if (strcmp(coll, "reducescatter") == 0) {
    algoMeta->sizeMultiplier = nGpus;
    algoMeta->func = mscclFuncReduceScatter;
  } else if (strcmp(coll, "allgather") == 0) {
    algoMeta->sizeMultiplier = nGpus;
    algoMeta->func = mscclFuncAllGather;
  } else if (strcmp(coll, "send") == 0) {
    algoMeta->func = mscclFuncSend;
  } else if (strcmp(coll, "recv") == 0) {
    algoMeta->func = mscclFuncRecv;
  } else if (strcmp(coll, "gather") == 0) {
    algoMeta->func = mscclFuncGather;
  } else if (strcmp(coll, "scatter") == 0) {
    algoMeta->func = mscclFuncScatter;
  } else if (strcmp(coll, "alltoall") == 0) {
    algoMeta->sizeMultiplier = nGpus;
    algoMeta->func = mscclFuncAllToAll;
  } else if (strcmp(coll, "alltoallv") == 0) {
    algoMeta->func = mscclFuncAllToAllv;
  } else {
    return ncclInvalidUsage;
  }

  int64_t minBytes;
  NCCLCHECK(mscclXmlGetAttrInt64(node, "minBytes", &minBytes));
  algoMeta->minBytes = minBytes;

  int64_t maxBytes;
  NCCLCHECK(mscclXmlGetAttrInt64(node, "maxBytes", &maxBytes));
  algoMeta->maxBytes = maxBytes;

  int inplace;
  NCCLCHECK(mscclXmlGetAttrInt(node, "inplace", &inplace));
  algoMeta->inPlace = (bool)inplace;

  int outofplace;
  NCCLCHECK(mscclXmlGetAttrInt(node, "outofplace", &outofplace));
  algoMeta->outOfPlace = (bool)outofplace;

  free(node);
  return ncclSuccess;
}
