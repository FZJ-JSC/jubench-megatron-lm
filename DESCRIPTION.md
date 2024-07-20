# NLP (Megatron-LM)

## Purpose

Megatron-LM is a widely used codebase in Natural Language Processing (NLP) for training Large Language Models. Using Megatron-LM a 175B parameter model can be trained using features such as flash-attention, distributed optimizers, sequence, tensor and pipeline parallelism. The goal of this benchmark is to measure the computational performance of a system repeating the training of this model. As the model is large, the number of nodes in virtually all configurations must be >> 1. This benchmark allows for choosing the smallest configuration (number of nodes) that can host this computation.

The implementation trains a GPT-3-type transformer model using a hybrid parallelization scheme. On the innermost level it uses tensor parallelism to parallelize the calculation of different attention heads and sequence parallelism for non-tensor parallel regions. It distributes the model between multiple GPUs with pipeline parallelism and uses data parallelism to scale to hundreds or thousands of GPUs. 

The parallelism scheme requires a certain relationship between the number of nodes, and model parameters, such as the number of layers or heads.In the reference setup on JUWELS Booster, this requires 96 nodes. For development, a version is provided that can run on 8 nodes. 

The benchmark measures the throughput of the training in tokens per unit time for a model of the given size. To be comparable to other benchmarks, the throughput is converted into a time by using a hypothetical number of tokens for which a training would be required to run. 

## Source

Archive name: `nlp-bench.tar.gz`

The file holds instructions to run the benchmark with or without JUBE, and
configuration files to run Megatron. The source of Megatron, and required library Apex are part of the archive; version and commits of the sources are the following:

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM): f7727433293427bef04858f67b2889fe9b177d88 (with `benchmark/aux/add_tflops_logging.patch` patch applied)
- [Apex](https://github.com/NVIDIA/apex): 8ffc901e50bbf740fdb6d5bccb17f66a6ec8604e

## Building

A path variable `NLP_BENCH_ROOT` needs to be set by doing `export NLP_BENCH_ROOT=<rootdir path of this folder>` in the calling shell.

Megatron_LM & Apex requires a number of custom kernels to be compiled as well other library dependencies that needs to be built. To build and compile, first a virtual environment is set up using `benchmark/env/setup_venv.sh` and then kernels are built with the script `src/compile_build.sh`. The `src/compile_build.sh` script assumes that the environment set up can be activated executing `source benchmark/env/activate.bash`.

Depending on the exact configuration of the software installation, the activation script can be adapted, or replaced by an empty dummy. 

### JUBE

When using JUBE, `NLP_BENCH_ROOT` variable is set automatically. The virtual environment is set up in the step `setup_venv` and step `build` takes care of building the benchmark. 

## Input Data

In `src/data`, the repository contains a subset (790 samples, 10 MB) of the small version of the [Oscar dataset](https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz) that is already pre-processed using `benchmark/aux/job_preprocess_data.sh` using GPT-2 tokenizers `benchmark/aux/tokenizers`.

## Execution

### Manual

The main script of this benchmark is `pretrain_gpt.py` from the Megatron-LM folder. As the number of parameters that need to be provided is large, we suggest modifying the script `benchmark/slurm/jobscript_175B.sbatch` for the actual execution. This script assumes a Slurm job scheduler and uses a small number of specific environment variables (`SLURM.*`) for configuring the run.

The script sets a number of variables that control the model to be trained, the parallelization scheme, optimizer parameters etc. For execution, this script can be adjusted to the technical requirements of a target system. However, all parameters controlling the training process must remain unchanged. To run the benchmark do `sbatch jobscript_175B.sbatch`. Refer to `README.md` for detailed instructions.

### JUBE

When using JUBE, `benchmark/jube/nlp_benchmark.yaml` contains all relevant definitions and steps of the workflow; it might need to be adapted for the system at hand.

The benchmark can be executed with JUBE by calling `jube run benchmark/jube/nlp_benchmark.yaml --tag 175`. The dependencies inside of the JUBE script take care to create the virtual environment and building the kernels and libraries, as specified above.

More information about the workflow with JUBE can be found in `README.md`.

## Configuration/Modification Rules and Guidelines

The benchmark aims at measuring the speed of a training of a 175B parameter model. The size of the trained language model is described by the following parameters: hidden dimension (`NHIDDEN`), number of layers (`NLAYERS`) and number of heads (`NHEADS`). For the 175B parameter model, they are fixed to 12288, 96, and 96 respectively. This is the actual benchmark size, and should not be modified. _In the compact 13B parameter version (only provided for convenience of testing), they are set to 5120, 40 and 32._

The Megatron-LM codebase uses a sophisticated parallelization scheme, that is controlled by the parameters indicating the tensor parallelism (`TP_SIZE`) and the pipeline parallelism (`PP_SIZE`). The code automatically infers the data parallel size from the number of nodes, the number of GPUs per node, and the parameters for tensor, pipe, and data parallelism. Furthermore, the structure of a single iteration can be controlled by the parameters micro batch size (`MICRO_BATCH_SIZE`) and gradient accumulation size (`GAS`). These parameters are free to optimize. In the optimization of these parameters, the global batch size is changed, which can be considered as an optimization parameter.

The numerical accuracy of the floating point operations must not be reduced compared to the baseline. Usually, the degree of parallelization is limited by the available GPU memory.

Per default, the execution of this benchmark requires the use of a released mainline version of PyTorch. With the submission, it is required to specify the commit hashes/versions of PyTorch and other software libraries used.  
If the run requires a PyTorch implementation that deviates from official releases, the implementation correctness must be proven with the submission (e.g. by executing a suited set of tests) and a path how the implementation will be integrated into PyTorch releases must be presented, preferably by example of past code contributions.

### Comments:

- PyTorch elastic launcher [torchrun](https://pytorch.org/docs/stable/elastic/run.html) is used for the benchmark. The launcher requires the following arguments for distributed training:

  ```
  torch.distributed.run \
      --nproc_per_node $GPUS_PER_NODE \
      --nnodes $NNODES \
      --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
      --rdzv_backend c10d \
      --max_restarts 0 \
      --tee 3
  ```
  where `MASTER_ADDR` is the FQDN of the host that is running worker with rank 0; used to initialize the Torch Distributed backend and `MASTER_PORT` is the port on the `MASTER_ADDR` that can be used to host the C10d TCP store. For JUWELS Booster, `MASTER_ADDR` is augmented with an additional `i` at the end for technical reasons.

  *Pro-Tip*: In case the following error is encountered
  ```
  torch.distributed.elastic.rendezvous.api.RendezvousConnectionError: The connection to the C10d store has failed. See inner exception for details.
  ```
  Please refer to this [issue](https://github.com/pytorch/pytorch/pull/81691) and make necessary patches to the PyTorch used.
- It was reported that when training large models, there could be a possibility of [training hanging](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles-prequel.md#hanging-issue ); the [fix](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles-prequel.md#dealing-with-jz-hanging-on-the-large-model) recommended is to set `CUDA_LAUNCH_BLOCKING=1`.
However, when using the `CUDA_LAUNCH_BLOCKING=1` fix, be careful to make sure that the NCCL version is NCCL/2.12.7-1 or older. For newer NCCL versions the execution hangs, due to some internal CUDA stream clashes.
- The following warning can be ignored if seen
  ```
  [W socket.cpp:558] [c10d] The client socket cannot be initialized to connect to [jwb0198i.juwels]:6000 (errno: 97 - Address family not supported by protocol).
  ```
- An evaluation version of the benchmark on 8 nodes for a 13B model training can be done by running `jube run benchmark/jube/nlp_benchmark.yaml --tag 13`

## Verification

If the application runs through successfully without any exceptions or error codes generated, then logging similar to the following will be seen in the `job.out` file created:

```
[default3]: iteration       10/ 3125000 | consumed samples:          960 | elapsed time per iteration (s): 7.2310 | learning rate: 4.438E-07 | global batch size:    96 | lm loss: 1.010967E+01 | loss scale: 4096.0 | grad norm: 178.553 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 13.276 | TFLOPs: 101.54 |
[default3]: iteration       20/ 3125000 | consumed samples:         1920 | elapsed time per iteration (s): 5.3551 | learning rate: 8.876E-07 | global batch size:    96 | lm loss: 8.861832E+00 | loss scale: 4096.0 | grad norm: 46.241 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 17.927 | TFLOPs: 137.10 |
[default3]: iteration       30/ 3125000 | consumed samples:         2880 | elapsed time per iteration (s): 5.6192 | learning rate: 1.331E-06 | global batch size:    96 | lm loss: 8.532865E+00 | loss scale: 4096.0 | grad norm: 8.121 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 17.084 | TFLOPs: 130.66 |
[default3]: iteration       40/ 3125000 | consumed samples:         3840 | elapsed time per iteration (s): 5.3291 | learning rate: 1.775E-06 | global batch size:    96 | lm loss: 8.442033E+00 | loss scale: 4096.0 | grad norm: 207.531 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 18.014 | TFLOPs: 137.77 |
[default3]: iteration       50/ 3125000 | consumed samples:         4800 | elapsed time per iteration (s): 5.3247 | learning rate: 2.219E-06 | global batch size:    96 | lm loss: 8.110270E+00 | loss scale: 4096.0 | grad norm: 5.662 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 18.029 | TFLOPs: 137.89 |

```

This indicates that the training run is performed. 

## Results

The standard output of the Megatron-LM is a throughput in "Tokens/sec". While very convenient in every day life, the metric of the benchmark is derived from this value to homogenize with other benchmarks. For submission, the throughput is converted into **time for a hypothetical training**.

The conversion from throughput to benchmark metric is done by assuming a training with 20 Million tokens, using the formula

```
[ time_to_report_in_seconds ] = [tokens] / [tokens/second]
```

This metric, calculated for the **175B model** , should be reported for submission (refer to Baseline Section below).

With JUBE, the result for the model can be generated using the command `jube result -a benchmark/jube/nlp_benchmark_run`. An example is shown in the following table (abbreviated):

| Model Size [Billion Param] | Nodes | Batch Size | Pipline Parallel | Tensor Parallel | Tokens/sec | Report Time |
|----------------------------|-------|------------|------------------|-----------------|------------|-------------|
|                         13 |     8 |       1024 |                4 |               2 |   60777.68 |     329.07   |
|                        175 |    96 |        96  |                8 |               8 |   36055.39 |     554.70   |

For the model to be trained in the benchmark (175B), the hypothetical training time of 20,000,000 / 36055.39 =  554.70  seconds is achieved. For evaluation purposed also given is a smaller model with 13B parameters; this is **not the reference model to report**. For JUWELS Booster, the distribution and parallelization scheme for the 175B model is `tp_size=8, pp_size=8, micro_batch_size=1, gas=16`, and for the 13B evaluation-only model is `tp_size=2, pp_size=4, micro_batch_size=1, gas=256`.

## Baseline

The baseline configuration of the benchmark must be chosen such that the virtual runtime of the 175B model for 20,000,000 tokens is less than or equal to 560 s. For JUWELS Booster, this is achieved for 96 nodes with 4 A100 GPUs each.

