import torch
import torch._C._te as te
from utils import *
scope = te.KernelScope()

# Sum 2nd dim
n, m = te_varhandle('n'), te_varhandle('m')
A = te_placeholder((n, m), name='a')
# hmmm, not sure about the reduction APIs.... I'm not sure if it's better or worse than TVM.
B = te.Reduce('b', get_dim_args([n]), te.Sum(), A, get_dim_args([m]))
s = te.LoopNest([B])
s.prepare_for_codegen()
mod = te_codegen('llvm', s.root_stmt(), [A, B, n, m])

a = torch.randn((3,4))
c = torch.empty((3,))
mod(a, c, 3, 4)
torch_assert_equal(a.sum(dim=1), c)

# Sum everything
B = te.Reduce('b', get_dim_args([]), te.Sum(), A, get_dim_args([n, m]))
s = te.LoopNest([B])
s.prepare_for_codegen()
mod = te_codegen('llvm', s.root_stmt(), [A, B, n, m])

c = torch.empty(())
mod(a, c, 3, 4)
torch_assert_equal(a.sum(), c)

# Custom Reduction
product = te.Reducer(to_expr(1.0), lambda a, b: a*b)
B = te.Reduce('b', get_dim_args([n]), product, A, get_dim_args([m]))
s = te.LoopNest([B])
s.prepare_for_codegen()
mod = te_codegen('llvm', s.root_stmt(), [A, B, n, m])
c = torch.empty((3,))
mod(a, c, 3, 4)
torch_assert_equal(a.prod(dim=-1), c)
