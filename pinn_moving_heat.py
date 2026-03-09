import jax
import jax.numpy as jnp
from jax import grad, vmap, hessian, jit
import equinox as eqx
import jinns
from jinns.parameters import Params
from jinns.loss import PDENonStatio, LossPDENonStatio, LossWeightsPDENonStatio
from jinns.data import CubicMeshPDENonStatio
import optax
import matplotlib.pyplot as plt

# --------------------------- 0. RNG ---------------------------
key = jax.random.PRNGKey(0)

# --------------------------- 1. Hyper-parameters ---------------------------
k  = 1e-5          # diffusivity
v  = 0.1           # source speed (x-direction)
sigma = 0.05
Q  = 1e-4
u0 = 300.0         # initial temperature

x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0
t_min, t_max = 0.0, 1.0

# --------------------------- 2. Network ---------------------------
eq_list = [
    (eqx.nn.Linear, 3, 64), (jax.nn.tanh,),
    (eqx.nn.Linear, 64, 64), (jax.nn.tanh,),
    (eqx.nn.Linear, 64, 64), (jax.nn.tanh,),
    (eqx.nn.Linear, 64, 1)               # scalar temperature
]

u_pinn, init_nn_params = jinns.nn.PINN_MLP.create(
    key=key, eqx_list=eq_list, eq_type="nonstatio_PDE"
)

params = Params(nn_params=init_nn_params, eq_params={})   # no learnable eq params
# --------------------------- 3. Forward helpers ---------------------------
@jit
def forward(x_t , params):
    """single point x_t : (3,) """
    """batch  : (N,3) """
    t = x_t[0:1]
    x = x_t[1:3]
    return u_pinn( t, x , params)  # JINNS wrapper = .apply

# ------------------------------------------------------------------
# 3-a) SINGLE POINT
# ------------------------------------------------------------------
xt_single = jnp.array([0.0, 0.0, 0.0])    # shape (1,3)
u_single = forward(xt_single, params)
# print("Single-point output shape :", u_single.shape)# (1,1)
# print("u(x=0,y=0,t=0) :", u_single)

# ------------------------------------------------------------------
# 3-b) BATCH
# ------------------------------------------------------------------
key_batch, key = jax.random.split(key)
xt_batch = jax.random.uniform(
  key_batch, shape=(10000, 3),
  minval=jnp.array([t_min,x_min, y_min]),
  maxval=jnp.array([t_max,x_max, y_max])
)

u_batch_fn = vmap(forward,in_axes=(0,None))
u_batch = u_batch_fn(xt_batch,params)
# print("Batch output shape :", u_batch.shape)          # (10000,1)

# --------------------------- 4. PDE loss (works on ANY batch size) ---------------------------
class HeatMovingSource(PDENonStatio):
  def __init__(self):
      super().__init__()

  # @eqx.filter_jit
  def equation(self, xt, u, params):
    """
    u      : PINN_MLP  (callable)
    xt     : (N,3) array  [t, x, y]
    params : Params
    """
    t, x, y = xt[:,0:1], xt[:,1:2], xt[:,2:3]
    # ---- debug
    # print(f'u_val shape: {u_val.shape}') # remove it when the bug is fixed 
    # for single point 
    def u_single_point(p):
      t = p[0:1]
      x = p[1:3]
      return u(t,x,params)[0,0]

    grad_u_batch = vmap(grad(u_single_point))(xt)
    # ---- ∂u/∂t -------------------------------------------------------
    du_dt = grad_u_batch[:, 0:1]    # (N,1)
    # ---- Laplacian ∇²u (hessian) ------------------------------------
    ## Hessian is expensive , replace Hessian with double derivative 
    '''
    hessian_u_single = hessian(u_single_point,argnums=0)
    hessian_u_batch = vmap(hessian_u_single)(xt)
    lap = (hessian_u_batch[:, 0, 0] + hessian_u_batch[:,1,1])[:, None]  # (N,1)
    '''
    def u_xx(p): return grad(lambda q: grad(u_single_point)(q)[1])(p)[1]
    def u_yy(p): return grad(lambda q: grad(u_single_point)(q)[2])(p)[2]
    lap = vmap(u_xx)(xt)[:,None] + vmap(u_yy)(xt)[:,None]

    # ---- moving Gaussian source --------------------------------------
    xc = x - v * t
    q = Q * jnp.exp(-(xc**2 + y**2) / (2 * sigma**2)) / (2 * jnp.pi * sigma**2)
    # ---- residual ----------------------------------------------------
    residual = du_dt - k * lap - q

    return residual # (N,1)

# ------------------------------------------------------------------
# 4-a) Test PDE on SINGLE point
# ------------------------------------------------------------------
pde = HeatMovingSource()
u_batch_test = vmap(forward,in_axes=(0,None))
res_batch = pde.equation(xt_batch, u_pinn , params)
print("\nPDE residual (single point) shape :", res_batch.shape)   # (1,1)

# ------------------------------------------------------------------
# 4-b) Test PDE on BATCH
# ------------------------------------------------------------------
res_batch = pde.equation(xt_batch, u_pinn, params)
# print("PDE residual (batch) shape :", res_batch.shape)            # (10000,1)

# --------------------------- 5. Tiny training loop (proof) ---------------------------
train_data = CubicMeshPDENonStatio(
  key=key,
  n=2000, nb=200, ni=50,
  dim=2,
  min_pts=(x_min, y_min),
  max_pts=(x_max, y_max),
  method="grid",
  tmin= t_min,
  tmax= t_max,
)

# Neumann ∂u/∂n = 0 → normal derivative = 0
def neumann(side):
  if side in ("xmin", "xmax"):
    idx = 1
    sign = -1 if side == "xmax" else 1
  else:                       # ymin / ymax
    idx = 2
    sign = -1 if side == "ymax" else 1
  def deriv(xt, params):
    def u_one(p):
        t = p[0:1]
        x = p[1:3]
        return u_pinn(t , x , params)[0]
    return sign * vmap(grad(u_one))(xt)[:, idx:idx+1]
  return deriv


loss = LossPDENonStatio(
    u=u_pinn,
    loss_weights=LossWeightsPDENonStatio(dyn_loss=1.0,
    initial_condition=10.0,
    boundary_loss=10.0),
    dynamic_loss=HeatMovingSource(),
    initial_condition_fun=lambda xt, p: jnp.full((xt.shape[0], 1), u0),
    omega_boundary_fun={s: neumann(s) for s in ("xmin","xmax","ymin","ymax")},
    omega_boundary_condition={s: "neumann" for s in ("xmin","xmax","ymin","ymax")},
    omega_boundary_dim={s: jnp.s_[0:1] for s in ("xmin","xmax","ymin","ymax")},
    params=params
)

opt = optax.adam(1e-3)
params_opt, losses, _ = jinns.solve(
  init_params=params,
  data=train_data,
  optimizer=opt,
  loss=loss,
  n_iter=2000,
  print_loss_every=500
)

plt.plot(losses)
plt.yscale('log')
plt.title('Training loss (2000 iters)')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
 