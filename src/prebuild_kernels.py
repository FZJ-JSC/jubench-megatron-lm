from argparse import Namespace
import os
from megatron import fused_kernels 

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
args = Namespace()
args.rank = 0
args.masked_softmax_fusion = True
fused_kernels.load(args)