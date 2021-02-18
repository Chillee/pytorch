import torch
import torch._C._te as te
from utils import *
scope = te.KernelScope()

def matmul(n, m, l):
    A = te_placeholder((n, l), name='A')
    B = te_placeholder((l, m), name='B')
    # TVM allows you to basically express the reduction in-place.
    mm = te_compute((n, m, l), lambda x, y, z: A.load([x,z]) * B.load([z, y]))
    C = te.Reduce('C', get_dim_args([n, m]), te.Sum(), mm, get_dim_args([l]))
    return A, B, C

n = 100
A, B, C = matmul(n, n, n)
s = te.LoopNest([C])
s.prepare_for_codegen()
print(te.simplify(s.root_stmt()))
mod = te_codegen('llvm', s.root_stmt(), [A, B, C])

a, b, c = get_abc((100, 100), torch.tensor)
mod(a, b, c)
torch_assert_equal(a @ b, c)



x = np.random.normal(size=4).astype("float32")
y = np.empty_like(x)
x, y = torch.tensor(x), torch.tensor(y)
n = te_varhandle("n")
A = te_placeholder((n,), name="A")
B = te_compute((n,), lambda i: A.load([i]), name="B")
s = te.LoopNest([B])
s.prepare_for_codegen()
mod = te_codegen("llvm", te.simplify(s.root_stmt()), [A, B, n])
mod(x, y,4)
print(x)