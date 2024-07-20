# JUPITER Benchmark Suite: Megatron-LM

[![DOI](https://zenodo.org/badge/831394735.svg)](https://zenodo.org/badge/latestdoi/831394735) [![Static Badge](https://img.shields.io/badge/DOI%20(Suite)-10.5281%2Fzenodo.12737073-blue)](https://zenodo.org/badge/latestdoi/764615316)

This benchmark is part of the [JUPITER Benchmark Suite](https://github.com/FZJ-JSC/jubench). See the repository of the suite for some general remarks.

This repository contains the Megatron-LM NLP/LLM benchmark. [`DESCRIPTION.md`](DESCRIPTION.md) contains details for compilation, execution, and evaluation.

The required source code (Megatron-LM, Apex) is included in the `./src/` subdirectory as submodules from the upstream repositories; [github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for Megatron-LM and [github.com/NVIDIA/apex](https://github.com/NVIDIA/apex) for Apex. Sample data files are also included.

## Overview of Benchmark

### Description Of Folder Structure

- benchmark
    * aux  
        - tokenizers
        - script used for getting data and tokenizers; `get_shrink_data_and_tokenizers.sh`
        - script used for preprocessing data; `job_preprocess_data.sbatch`
        - sample 10MB oscar dataset got using `get_shrink_data_and_tokenizers.sh`
    * env
        - script for activating the python virtual env; `activate.bash`
        - script to set up python virtual env; `setup_venv.sh`
    * slurm 
        - sbatch scripts for 13B and 175B model to be used when running without JUBE
    * jube
        - contains accompanying files for JUBE run and the JUBE yaml file
- src
    * data : contains the preprocessed data (`*idx` and `*.bin` files)
    * `compile_build.sh` : script to build the software dependencies
    * `variables.bash` : file that sets important paths
    * `prebuild_kernels.py` : script to prebuild fused kernels
    

### Workflow Without JUBE:

#### Getting Data and Tokenizers

The following workflow can be done if data and tokenizers are not already present with this repository:

- __*Step 1*__: Set `NLP_BENCH_ROOT` variable as `export NLP_BENCH_ROOT=<rootdir path of this benchmark>` in your bash shell
- __*Step 2*__: `cd benchmark/aux/`
- __*Step 3*__: `bash get_shrink_data_and_tokenizers.sh` to get tokenizers and compress the raw data `oscar-1GB.jsonl.xz` to `oscar-10MB.jsonl.xz`

#### Prepocessing Data

If your `src/data` folder does not contain preprocessed data (`*.idx` and `*.bin` files), then execute 
`sbatch job_preprocess_data.sbatch` after _Step 5_ in "Workflow With Preprocessed Data And Tokenizers Available" from `benchmark/aux` directory.

The `job_preprocess_data.sbatch` script in `benchmark/aux/` is used to preprocess the `oscar-10MB.jsonl.xz` and put it in `src/data/`. The file can be modified to preprocess any data of choice.

#### Workflow With Preprocessed Data And Tokenizers Available

- __*Step 1*__: `cd` into it the folder of this benchmark
- __*Step 2*__: Set `NLP_BENCH_ROOT` variable as `export NLP_BENCH_ROOT=<rootdir path of this benchmark>` in your bash shell
- __*Step 3*__: Set `TORCH_CUDA_ARCH_LIST` according to GPU's compute capability in `benchmark/env/activate.bash`
- __*Step 4*__: Run `bash benchmark/env/setup_venv.sh`
- __*Step 5*__: Run `bash src/compile_build.sh`
- __*Step 6*__: Run `sbatch benchmark/slurm/jobscript_13B.sbatch ` or `sbatch benchmark/slurm/jobscript_175B.sbatch`

The output file `*.out` file would have result logs of the following form that are important :
```
[default3]: iteration       10/  292968 | consumed samples:        10240 | elapsed time per iteration (s): 35.8651 | learning rate: 4.734E-06 | global batch size:  1024 | lm loss: 1.332803E+01 | loss scale: 4096.0 | grad norm: 42.627 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 28.551 | TFLOPs: 199.03 |
[default3]: iteration       20/  292968 | consumed samples:        20480 | elapsed time per iteration (s): 34.9991 | learning rate: 9.467E-06 | global batch size:  1024 | lm loss: 1.010884E+01 | loss scale: 4096.0 | grad norm: 13.038 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 29.258 | TFLOPs: 203.96 |
[default3]: iteration       30/  292968 | consumed samples:        30720 | elapsed time per iteration (s): 34.8709 | learning rate: 1.420E-05 | global batch size:  1024 | lm loss: 9.072961E+00 | loss scale: 4096.0 | grad norm: 26.640 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 29.365 | TFLOPs: 204.71 |
[default3]: iteration       40/  292968 | consumed samples:        40960 | elapsed time per iteration (s): 35.3346 | learning rate: 1.893E-05 | global batch size:  1024 | lm loss: 8.486469E+00 | loss scale: 4096.0 | grad norm: 3.441 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 28.980 | TFLOPs: 202.02 |
[default3]: iteration       50/  292968 | consumed samples:        51200 | elapsed time per iteration (s): 35.3357 | learning rate: 2.367E-05 | global batch size:  1024 | lm loss: 8.

```

The metric `tokens_per_sec` should be calculated as `(1.0/$elapsed_time_per_iteration)*$global_batch_size*$sequence_length` obtained from the `*.out` file.

For submission the throughput tokens_per_sec is converted into **time, a hypothetical training would require**. This conversion is done by assuming a training with 20 Million tokens, using the formula
```
[ time_to_report_in_seconds ] =  [tokens] / [tokens/second] 
```
Example: Using the 13B model result below (Tokens/sec:  59463.14), we obtain a duration of 20,000,000 /  59463.14 = 336.34 seconds.

__*Hint*__: sequence_length can be found in the jobscript.

### Workflow With JUBE:

- __*Step 1*__: `cd` into it the folder of this benchmark
- __*Step 2*__: Set `TORCH_CUDA_ARCH_LIST` according to GPU's compute capability in `benchmark/env/activate.bash`
- __*Step 3*__: Execute either `jube run benchmark/jube/nlp_benchmark.yaml --tag 175` for 175B model or`jube run benchmark/jube/nlp_benchmark.yaml --tag 13` for 13B model
- __*Step 4*__: Wait for the benchmark to run and then do `jube continue nlp_benchmark_run -i last` until no *Step*s with the "wait" state remain
- __*Step 5*__: After the benchmark finishes, run `jube result -a nlp_benchmark_run -i last` to print the benchmark results

Example result from JUBE:

```
|        system | version |   queue |    JobID |   Job_Time | Model_Size (Billion Param) | Nodes | Batch_Size | Pipeline_Parallel | Tensor_Parallel | Iterations | Avg_TFLOPs/GPU | Tokens/sec | time_to_report_in_seconds |
|---------------|---------|---------|----------|------------|----------------------------|-------|------------|-------------------|-----------------|------------|----------------|------------|---------------------------|
| juwelsbooster | 2024.01 | booster | 10011638 | "00:30:00" |                         13 |     8 |       1024 |                 4 |               2 |         20 |        206.885 |   60777.68 |                    329.07 |

```
