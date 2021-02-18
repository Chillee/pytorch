import torch
import torch._C._te as te
from utils import *
scope = te.KernelScope()

# Transpose
n = te_varhandle('n')
m = te_varhandle('m')
A = te_placeholder((n, m), name='a')
B = te_compute((m, n), lambda i, j: A.load([j, i]), 'b')
s = te.LoopNest([B])
s.prepare_for_codegen()
mod = te_codegen('llvm', s.root_stmt(), [A, B, n, m])

a = np.arange(12, dtype='float32').reshape((3, 4))
b = np.empty((4, 3), dtype='float32')
a, b = torch.tensor(a), torch.tensor(b)

mod(a, b, 3, 4)
print(a)
print(b)


# General reshape
p, q = te_varhandle('p'), te_varhandle('q')
B = te_compute((p, q), lambda i, j: A.load([(i*q+j)/m, (i*q+j)%m]), 'b')
s = te.LoopNest([B])
s.prepare_for_codegen()
mod = te_codegen('llvm', s.root_stmt(), [A, B, n, m, p, q])
b = np.zeros((5, 4), dtype='float32')
a, b = torch.tensor(a), torch.tensor(b)
mod(a,b, 3,4,5,4)
print(a)
print(b)

# Slicing
bi, bj, si, sj = [te_varhandle(name) for name in ['bi', 'bj', 'si', 'sj']]
B = te_compute(((n-bi)/si, (m-bj)/sj), lambda i, j: A.load([i*si+bi, j*sj+bj]), 'b')
s = te.LoopNest([B])
s.prepare_for_codegen()
mod = te_codegen('llvm', s.root_stmt(), [A, B, n, m, bi, si, bj, sj])
b = torch.empty((1, 3))
mod(a, b, 3, 4, 1, 2, 1, 1)
torch_assert_equal(b, a[1::2, 1::1])
b = torch.empty((1, 2))
mod(a, b, 3, 4, 2, 1, 0, 2)
torch_assert_equal(b, a[2::1, 0::2])

