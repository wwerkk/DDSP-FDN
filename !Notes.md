# Remarks
@katz: 
- seems like its generally a difficult problem to model IIRs
- preferred pipeline could be a recorded IR -> parameters for a reverb (Schroeder/Jot) approximating such response


Making the parameters of the Schroeder reverb differentiable for use in neural network training is a non-trivial task. The main challenge is that the Schroeder reverb algorithm involves operations that are not naturally differentiable, such as delay lines and feedback loops. However, there are a few strategies you could use:

1. **Approximate the derivatives:** You could use numerical methods to approximate the derivatives of the parameters with respect to the output. This would allow you to use gradient-based optimization methods, but the approximations may not be accurate, especially for parameters that affect the system in complex ways.

2. **Use differentiable approximations of the operations:** For example, you could replace the delay lines with a differentiable approximation, such as a fractional delay filter. This would make the system differentiable, but it may not accurately represent the original Schroeder reverb.

3. **Use reinforcement learning:** Instead of trying to make the Schroeder reverb differentiable, you could use reinforcement learning to train the parameters. This would involve defining a reward function that measures the quality of the reverb and using an algorithm like Q-learning or policy gradients to optimize the parameters. This approach can handle non-differentiable systems, but it may require a lot of data and computational resources.

4. **Use a differentiable surrogate model:** Another approach is to train a differentiable surrogate model that approximates the Schroeder reverb. The surrogate model could be a neural network that is trained to mimic the output of the Schroeder reverb for a given set of parameters. Once the surrogate model is trained, it can be used for gradient-based optimization of the parameters.

Each of these strategies has its own trade-offs and challenges, and the best approach may depend on the specific requirements of your application. It's also worth noting that this is an active area of research, and there may be other strategies that I haven't mentioned here.