import torch
import torch._C._te as te
from utils import *
scope = te.KernelScope()

def te_varhandle(name=None, dtype=torch.int):
    if name is None:
        name = 'n'
    return te.VarHandle(name, get_nnc_type(dtype))

n = te_varhandle('n')
A = te_placeholder((n,), name='a')
B = te_placeholder((n,), name='b')
C = te_compute((n,), lambda i: A.load([i]) + B.load([i]), name='c')
s = te.LoopNest([C])
# We should be able to infer that the variable length is part of the input tensors, so we don't need to explicitly pass it in.
# Also, it should throw an error if it's never passed in lol
mod = te_codegen('llvm', s.root_stmt(), [A,B,C, n])

def test_mod(mod, n):
    # zzz how do we make the pybind bindings do this properly?
    a, b, c = get_abc(n, torch.tensor)
    mod(a,b,c, *[torch.tensor(i) for i in n])
    np.testing.assert_equal(c.numpy(), (a+b).numpy())

test_mod(mod, (5,))
test_mod(mod, (100,))


def tvm_vector_add(ndim):
    shapes = [te_varhandle() for _ in range(ndim)]
    A = te_placeholder(shapes)
    B = te_placeholder(shapes)
    C = te_compute(shapes, lambda *i: A.load([*i]) + B.load([*i]))
    s = te.LoopNest([C])
    # This really shouldn't be necessary...
    s.prepare_for_codegen()
    return te_codegen('llvm', s.root_stmt(), [A,B,C] + shapes)

mod = tvm_vector_add(2)
test_mod(mod, (2, 2))
mod = tvm_vector_add(4)
test_mod(mod, (2, 3,4,5))