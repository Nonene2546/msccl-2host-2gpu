from msccl.language import *
from msccl.topologies import fully_connected
from msccl.language.collectives import AllGather


def allgather_2gpu():
    # 2 ranks, 1 chunk each. Rank r sends its input chunk to the other rank.
    for r in range(2):
        c = chunk(r, Buffer.input, 0)
        other = 1 - r
        c.copy(other, Buffer.output, r, sendtb=0, recvtb=0)
        # Also copy own chunk to its output slot.
        c.copy(r, Buffer.output, r, sendtb=0, recvtb=0)


topo = fully_connected(2)
coll = AllGather(2, 1, inplace=False)
with MSCCLProgram(
    "allgather_2gpu",
    topo,
    coll,
    instances=1,
    protocol="Simple",
    threadblock_policy=ThreadblockPolicy.manual,
    instr_fusion=True,
):
    allgather_2gpu()
    print(Check())
    XML()
