import torch
import torch._C._te as te
from utils import *
scope = te.KernelScope()

# Take lower triangular matrix
n, m = te_varhandle('n'), te_varhandle('m')
A = te_placeholder((n, m), name='a')
# I think we should consider doing this mapping from values to ExprHandles.
B = te_compute((n, m), lambda i, j: te.ifThenElse(i >= j, A.load([i, j]), to_expr(0.0)))
s = te.LoopNest([B])
s.prepare_for_codegen()
mod = te_codegen('llvm', s.root_stmt(), [A, B, n, m])
a = torch.randn(3, 4)
b = torch.empty(3, 4)
mod(a, b, 3, 4)
print(b)

# Add padding
p = to_expr(1) # padding size
# Perhaps adding a `te.any` or `te.all` would be nice...
B = te_compute((n+p*to_expr(2), m+p*to_expr(2)),
                lambda i, j: te.ifThenElse( (i<p) | (i>=n+p) | (j<p) | (j>=m+p), to_expr(0.0), A.load([i-p, j-p])), name='b')
s = te.LoopNest([B])
s.prepare_for_codegen()
mod = te_codegen('llvm', s.root_stmt(), [A, B, n, m])
a = torch.randn(3, 4)
b = torch.empty(5, 6)
mod(a, b, 3, 4)
print(b)

