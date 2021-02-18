import torch._C._te as te
import torch
import numpy as np
import timeit
from IPython import display
from matplotlib import pyplot as plt



def torch_assert_equal(a, b):
    return np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-5)

def get_te_shapes(shape):
    return [te.ExprHandle.int(i) if isinstance(i, int) else i for i in shape]

def timer_iters(f):
    return lambda it: timeit.timeit(f, number=it)

def get_nnc_type(dtype):
    if dtype == torch.float:
        return te.Dtype.Float
    elif dtype == torch.long:
        return te.Dtype.Long
    elif dtype == torch.int:
        return te.Dtype.Int
    elif dtype == torch.double:
        return te.Dtype.Double
    else:
        raise RuntimeError("nyi")

def get_dim_args(dims):
    dim_args = []
    for dim in get_te_shapes(dims):
        dim_args.append(te.DimArg(dim, chr(ord('a') + len(dim_args))))
    return dim_args

def to_expr(x):
    if isinstance(x, int):
        return te.ExprHandle.int(x)
    elif isinstance(x, float):
        return te.ExprHandle.float(x)

def te_varhandle(name=None, dtype=torch.int):
    if name is None:
        name = 'n'
    return te.VarHandle(name, get_nnc_type(dtype))

def te_placeholder(shape, dtype=torch.float, name=None):
    if name is None:
        name = 'a'
    return te.Placeholder(name, get_nnc_type(dtype), get_te_shapes(shape))

def te_compute(shape, f, name=None):
    if name is None:
        name = 'a'
    return te.Compute(name, get_dim_args(shape), f)

def te_codegen(backend, stmt, io_tensors):
    mod = te.construct_codegen(backend, stmt, [te.BufferArg(i) for i in io_tensors])
    def wrap(*args):
        args = [torch.tensor(i) if isinstance(i, int) else i for i in args]
        return mod.call(args)
    return wrap

def get_abc(shape, constructor=None):
    """Return random a, b and empty c with the same shape.
    """
    a = torch.randn(*shape).numpy()
    b = torch.randn(*shape).numpy()
    c = torch.empty(*shape).numpy()
    if constructor:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c

def get_bcast_data(shape1, shape2, constructor=None):
    """Return random tensors a, b
    and empty tensor c to store broadcast results between a and b

    shape1, shape2: shapes of input tensors
    constructor : user-defined tensor constructor
    """
    a = torch.randn(*shape1).numpy()
    b = torch.randn(*shape2).numpy()
    out_shape = (shape1[0] if shape2[0] == 1 else shape2[0],
                 shape1[1] if shape2[1] == 1 else shape2[1])
    c = torch.empty(*out_shape).numpy()
    if constructor:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c

def bench_workload(workload):
    """Benchmark a workload

    workload: a method that accept a num_repeat argument
    and return its total execution time
    """
    workload(1)  # warmup
    time = workload(1)  # the time to run once
    if time > 1: return time
    # The number of repeats to measure at least 1 second
    num_repeats = max(int(1.0 / time), 5)
    return workload(num_repeats) / num_repeats

def plot(X, Y, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear', fmts=None,
         figsize=(6, 4)):
    """Plot multiple lines"""
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
    axes = plt.gca()
    X, Y = np.array(X), np.array(Y)
    if X.shape != Y.shape: X = [X] * len(Y)
    if not fmts: fmts = ['-'] * len(X)
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend: axes.legend(legend)
    axes.grid()

def plot_gflops(sizes, gflops, legend, xlabel='Size'):
    plot(sizes, gflops, xlabel=xlabel, ylabel='GFLOPS',
             xscale='log', yscale='log',
             legend=legend, fmts=['--']*(len(gflops)-1)+['-'])