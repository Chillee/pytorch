# Save to the d2ltvm package.
def get_abc(shape, constructor=None):
    """Return random a, b and empty c with the same shape.
    """
    np.random.seed(0)
    a = np.random.normal(size=shape).astype(np.float32)
    b = np.random.normal(size=shape).astype(np.float32)
    c = np.empty_like(a)
    if constructor:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c

import torch
import torch._C._te as te
import numpy as np
scope = te.KernelScope()

def get_te_shapes(shape):
    return [te.ExprHandle.int(i) for i in shape]

def get_nnc_type(dtype):
    if dtype == torch.float:
        return te.Dtype.Float
    elif dtype == torch.long:
        return te.Dtype.Long
    else:
        raise RuntimeError("nyi")

def get_dim_args(dims):
    dim_args = []
    for dim in dims:
        dim_args.append(te.DimArg(te.ExprHandle.int(dim), 'i' + str(len(dim_args))))
    return dim_args

# Constructing a placeholder with input shapes is too hard. We should be able to just pass in a static shape.
# Perhaps there should be a default dtype too?
def te_placeholder(name, shape, dtype=torch.float):
    return te.Placeholder(name, get_nnc_type(dtype), get_te_shapes(shape))

def vector_add(n):
    A = te_placeholder(name='a', shape=(n,))
    B = te_placeholder(name='b', shape=(n,))
    # TVM allows you to query shape of already computed te.Tensor values
    # Do we really need to call A.load(i) or can we use indexing notation?
    C = te.Compute('c', get_dim_args((n,)), lambda i: A.load([i]) + B.load([i]))
    return A, B, C

A, B, C = vector_add(100)

# Is there an equivalent of Tensor.op?
# The operation that generates the tensor object can be accessed by A.op.
# type(A.op), type(C.op)
# (tvm.te.tensor.PlaceholderOp, tvm.te.tensor.ComputeOp)
print(A,B,C)

# TVM makes you call "create_schedule" (i.e. "LoopNest" in NNC?) to create a schedule. There seems to be some disentanglement in TVM between "making a schedule" and "applying a schedule". Not so in NNC
s = te.LoopNest([C])
s.prepare_for_codegen()
print(s)

# TVM has a concept of "stages" - what's the equivalent in NNC?

# Why do we need to wrap the values in BufferArgs? Shouldn't we able to do that automatically?
cg = te.construct_codegen('llvm', te.simplify(s.root_stmt()), [te.BufferArg(A), te.BufferArg(B), te.BufferArg(C)])

a, b, c = get_abc(100, torch.tensor)

cg.call([a, b, c])
np.testing.assert_array_equal((a + b).numpy(), c)


# If we provide invalid shapes to cg.call, TVM will crash but we'll silently break.
a, b, c = get_abc(200, torch.tensor)

cg.call([a, b, c])

# A compiled TVM module can be written/saved to disk, which is kinda neat.
