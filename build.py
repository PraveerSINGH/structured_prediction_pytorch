import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['algorithms/src/my_lib_cuda.c']
headers = ['algorithms/src/my_lib_cuda.h']
defines = [('WITH_CUDA', None)]
with_cuda = True

ffi = create_extension(
    'algorithms._ext.my_lib',
    package=True,
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda
)

if __name__ == '__main__':
    ffi.build()