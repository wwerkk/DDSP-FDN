from jax import jit
import jax.numpy as jnp
								
# define the cube function
def cube(x):
	return x * x * x

# generate data
x = jnp.ones((10000, 10000))

jit_cube = jit(cube)