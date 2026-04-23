# MSCCL Install — 2 Hosts × 2 GPUs

Installs the MSCCL-patched NCCL runtime and `nccl-tests` to run the included
XML schedule on a 4-rank cluster (2 hosts × 2 GPUs, 1 rank per GPU).

**Shipped:**
- `result-2host-2gpu-pcie-ag-0-Simple-fuse=False-depnop=False-i=1-inplace=False.xml` — main schedule: 1 MiB AllGather, PCIe-only 2 host × 2 GPU
- `allgather_2gpu.xml` — smoke-test schedule for 1 host × 2 GPU (see §6a)
- `gen_2gpu_ag.py` — msccl DSL source that generated `allgather_2gpu.xml`
- `syccl-config/2host-2gpu-pcie-ag.json` — SyCCL config that generated `result-2host-2gpu-pcie-ag-0-Simple-fuse=False-depnop=False-i=1-inplace=False.xml`

Run every section on **both hosts** (or once on shared FS).

> **Status:** Build steps (§1–§4) tested on a fresh conda env. Runtime steps
> (§5–§6) **not yet tested** — the author has only 1 GPU. The XML schedules
> pass static checks but haven't been run on real hardware.

---

## 0. Prerequisites

- NVIDIA driver working (`nvidia-smi` shows 2 GPUs per host)
- Conda / Miniconda installed
- Passwordless SSH from launcher → both hosts
- Inter-host network reachability (TCP; IB/RoCE if available)

---

## 1. Conda env

```bash
conda create -n msccl -c nvidia -c conda-forge \
    "cuda-version=12.4" "cuda-toolkit=12.4" \
    openmpi=5 python=3.10 cmake make \
    "gcc_linux-64=12" "gxx_linux-64=12" -y

conda activate msccl
export CUDA_HOME=$CONDA_PREFIX
export MPI_HOME=$CONDA_PREFIX
```

Verify:

```bash
nvcc --version | grep "release 12.4"
$CXX --version | grep "gcc 12"
mpirun --version | head -1
ls $CUDA_HOME/include/cuda_runtime.h
```

---

## 2. Clone repos

```bash
git clone https://github.com/Azure/msccl-executor-nccl.git ~/msccl
git clone https://github.com/NVIDIA/nccl-tests.git ~/nccl-tests
(cd ~/nccl-tests && git checkout v2.13.10)
```

---

## 3. Build MSCCL

Patch the Makefile:

```bash
cd ~/msccl
sed -i 's/-Xfatbin -compress-all //' makefiles/common.mk
```

Default gencode below targets **RTX 40-series**. For other GPUs, look up your
compute capability at <https://developer.nvidia.com/cuda-gpus> and substitute
`compute_XX,sm_XX` (plus PTX fallback `compute_XX,compute_XX`).

Build (use `-j2` or `-j1` if RAM ≤ 32 GB):

```bash
cd ~/msccl
make -j$(nproc) src.build \
    CUDA_HOME=$CUDA_HOME \
    NVCC_GENCODE="-gencode=arch=compute_89,code=sm_89 \
                  -gencode=arch=compute_89,code=compute_89"
```

Produces `~/msccl/build/lib/libnccl.so.2.23.4`.

---

## 4. Build nccl-tests

```bash
cd ~/nccl-tests
make -j$(nproc) MPI=1 \
    CUDA_HOME=$CUDA_HOME \
    MPI_HOME=$MPI_HOME \
    NCCL_HOME=$HOME/msccl/build \
    NVCC_GENCODE="-gencode=arch=compute_89,code=sm_89 \
                  -gencode=arch=compute_89,code=compute_89"
```

Produces `~/nccl-tests/build/all_gather_perf`.

---

## 5. Install the XML

```bash
mkdir -p ~/msccl-algos
cp result-2host-2gpu-pcie-ag-0-Simple-fuse=False-depnop=False-i=1-inplace=False.xml ~/msccl-algos/
cp allgather_2gpu.xml      ~/msccl-algos/   # for §6a smoke test
```

MSCCL loads every XML in the directory and matches each one to a comm by
`ngpus` / `coll` / size. You can keep both files there — only the matching one
fires for a given run.

---

## 6a. Smoke test — single host, 2 GPUs

Run this on **one** of the hosts to verify MSCCL loads and executes an XML
locally before tackling the 2-host setup. `mpirun` launches both ranks on the
same machine — no SSH, no `NCCL_SOCKET_IFNAME`, no inter-host networking.

```bash
conda activate msccl
export LD_LIBRARY_PATH=$HOME/msccl/build/lib:$LD_LIBRARY_PATH
export MSCCL_ALGO_DIR=$HOME/msccl-algos
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL

mpirun --allow-run-as-root -np 2 \
    -x LD_LIBRARY_PATH -x MSCCL_ALGO_DIR -x NCCL_ALGO \
    -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
    $HOME/nccl-tests/build/all_gather_perf \
        -b 1M -e 1M -f 2 -g 1 -c 1 -n 20
```

If the perf table prints `# Out of bounds values : 0 OK`, the runtime is wired
up correctly and you can move on to §6.

---

## 6b. Run — 2 hosts × 2 GPUs

```bash
conda activate msccl
export LD_LIBRARY_PATH=$HOME/msccl/build/lib:$LD_LIBRARY_PATH
export MSCCL_ALGO_DIR=$HOME/msccl-algos
export NCCL_ALGO=MSCCL,RING,TREE
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL
export NCCL_SOCKET_IFNAME=eth0   # check with `ip -br addr`

mpirun --allow-run-as-root \
    -H hostA:2,hostB:2 -np 4 \
    --map-by ppr:2:node --bind-to none \
    -x LD_LIBRARY_PATH -x MSCCL_ALGO_DIR -x NCCL_ALGO \
    -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_SOCKET_IFNAME \
    $HOME/nccl-tests/build/all_gather_perf \
        -b 1M -e 1M -f 2 -g 1 -c 1 -n 20
```

