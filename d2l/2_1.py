import torch
import torch._C._te as te
from utils import *
scope = te.KernelScope()

n = 100


def nnc_vector_add(dtype):
    # We should be able to have a default name.
    A = te_placeholder((n,), dtype=dtype)
    B = te_placeholder((n,), dtype=dtype)
    C = te_compute((n,), lambda i: A.load([i]) + B.load([i]))

    s = te.LoopNest([C])
    return te_codegen('llvm', s.root_stmt(), [A,B,C])



def test_dtype(dtype):
    a, b, c = get_abc(n, lambda x: torch.tensor(x, dtype=dtype))
    print('tensor dtype:', a.dtype, b.dtype, c.dtype)
    # It should probably be wrapped as a callable
    mod = nnc_vector_add(dtype)
    mod(a, b, c)
    np.testing.assert_equal(c.numpy(), (a+b).numpy())

for dtype in [torch.long, torch.float, torch.double]:
    test_dtype(dtype)

def nnc_vector_add_2(dtype):
    A = te_placeholder((n,))
    B = te_placeholder((n,))
    # Casting should ideally be a method of the tensor itself.
    C = te_compute((n,), lambda i: te.Cast.make(get_nnc_type(dtype), A.load([i])) + te.Cast.make(get_nnc_type(dtype), B.load([i])))

    s = te.LoopNest([C])
    return te_codegen('llvm', s.root_stmt(), [A,B,C])

def test_dtype2(dtype):
    a, b, c = get_abc(n, torch.tensor)
    c = c.to(dtype)
    print('tensor dtype:', a.dtype, b.dtype, c.dtype)
    # It should probably be wrapped as a callable
    mod = nnc_vector_add_2(dtype)
    mod(a, b, c)
    np.testing.assert_equal(c.numpy(), (a.to(dtype)+b.to(dtype)).numpy())

test_dtype2(torch.long)

