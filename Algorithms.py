import numpy as np

class TamedSGHMC:
	def __init__(self, H, lr, gamma, beta, m, r):
		# H: stochastic gradient, H(theta, x)
		# lr: learning rate (step size)
		# gamma: friction coefficient
		# beta: inverse temperature
		# m: strong convexity coefficient
		# r: taming exponent
		self.H = H
		self.lr = float(lr)
		self.gamma = float(gamma)
		self.beta = float(beta)
		self.m = float(m)
		self.r = float(r)

		if self.lr <= 0:
			raise ValueError("lr must be positive")
		if self.gamma <= 0:
			raise ValueError("gamma must be positive")
		if self.beta <= 0:
			raise ValueError("beta must be positive")
		if self.r < 0:
			raise ValueError("r must be non-negative")

	def _f(self, theta, x):
		return self.H(theta, x) - self.m * theta

	def _taming_denominator(self, theta):
		theta_norm = np.linalg.norm(theta)
		return np.sqrt(1.0 + (theta_norm ** (4.0 * self.r)) / self.gamma)

	def H_gamma(self, theta, x):
		return self.m * theta + self._f(theta, x) / self._taming_denominator(theta)

	def step(self, theta, vol, x, rng=None):
		if rng is None:
			rng = np.random.default_rng()

		theta = np.asarray(theta, dtype=float)
		vol = np.asarray(vol, dtype=float)

		noise_scale = np.sqrt(2.0 * self.lr * self.gamma / self.beta)
		xi = rng.normal(0, 1, size=theta.shape)

		vol_next = vol - self.lr * (self.gamma * vol + self.H_gamma(theta, x)) + noise_scale * xi
		theta_next = theta + self.lr * vol_next

		return theta_next, vol_next

	def simulate(self, theta0, vol0, xs, n_steps=None, rng=None):
		# theta0: initial position
        # vol0: initial velocity
		# xs: sequence of data points (or mini-batches)
		# n_steps: number of steps to simulate (if None, simulate for all xs)
		if rng is None:
			rng = np.random.default_rng()

		theta = np.asarray(theta0, dtype=float)
		vol = np.asarray(vol0, dtype=float)
		xs_iter = iter(xs)

		thetas = [theta.copy()]
		vols = [vol.copy()]

		step_count = 0
		for x in xs_iter:
			theta, vol = self.step(theta, vol, x, rng=rng)
			thetas.append(theta.copy())
			vols.append(vol.copy())
			step_count += 1
			if n_steps is not None and step_count >= n_steps:
				break

		return np.array(thetas), np.array(vols)


class SGHMC:
	"""
	Stochastic Gradient Hamiltonian Monte Carlo (non-tamed).

	Update rule:
		v_{n+1} = v_n - lr * (gamma * v_n + H(theta_n, x_{n+1}))
		          + sqrt(2 * lr * gamma / beta) * xi_{n+1}
		theta_{n+1} = theta_n + lr * v_{n+1}
	"""

	def __init__(self, H, lr, gamma, beta):
		# H: stochastic gradient, H(theta, x)
		# lr: learning rate (step size)
		# gamma: friction coefficient
		# beta: inverse temperature
		self.H = H
		self.lr = float(lr)
		self.gamma = float(gamma)
		self.beta = float(beta)

		if self.lr <= 0:
			raise ValueError("lr must be positive")
		if self.gamma <= 0:
			raise ValueError("gamma must be positive")
		if self.beta <= 0:
			raise ValueError("beta must be positive")

	def step(self, theta, vol, x, rng=None):
		if rng is None:
			rng = np.random.default_rng()

		theta = np.asarray(theta, dtype=float)
		vol = np.asarray(vol, dtype=float)

		noise_scale = np.sqrt(2.0 * self.lr * self.gamma / self.beta)
		xi = rng.normal(0.0, 1.0, size=theta.shape)

		vol_next = vol - self.lr * (self.gamma * vol + self.H(theta, x)) + noise_scale * xi
		theta_next = theta + self.lr * vol_next
		return theta_next, vol_next

	def simulate(self, theta0, vol0, xs, n_steps=None, rng=None):
		# theta0: initial position
		# vol0: initial velocity
		# xs: sequence of stochastic samples
		if rng is None:
			rng = np.random.default_rng()

		theta = np.asarray(theta0, dtype=float)
		vol = np.asarray(vol0, dtype=float)
		xs_iter = iter(xs)

		thetas = [theta.copy()]
		vols = [vol.copy()]

		step_count = 0
		for x in xs_iter:
			theta, vol = self.step(theta, vol, x, rng=rng)
			thetas.append(theta.copy())
			vols.append(vol.copy())
			step_count += 1
			if n_steps is not None and step_count >= n_steps:
				break

		return np.array(thetas), np.array(vols)

class TUSLA:
    def __init__(self, H, lr, beta, r):
        # H: stochastic gradient H(theta, x)
        # lr: step size lambda > 0
        # beta: inverse temperature > 0
        # r: taming exponent >= 0
        self.H = H
        self.lr = float(lr)
        self.beta = float(beta)
        self.r = float(r)

        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.r < 0:
            raise ValueError("r must be non-negative")

    def _taming_denominator(self, theta):
        theta_norm = np.linalg.norm(theta)
        return 1.0 + np.sqrt(self.lr) * (theta_norm ** (2.0 * self.r))

    def H_lambda(self, theta, x):
        return self.H(theta, x) / self._taming_denominator(theta)

    def step(self, theta, x, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        theta = np.asarray(theta, dtype=float)

        noise_scale = np.sqrt(2.0 * self.lr / self.beta)
        xi = rng.normal(0.0, 1.0, size=theta.shape)

        theta_next = theta - self.lr * self.H_lambda(theta, x) + noise_scale * xi
        return theta_next

    def simulate(self, theta0, xs, n_steps=None, rng=None):
        # theta0: initial parameter
        # xs: iterable of stochastic samples / mini-batches
        # n_steps: if set, stop after n_steps updates
        if rng is None:
            rng = np.random.default_rng()

        theta = np.asarray(theta0, dtype=float)
        xs_iter = iter(xs)

        thetas = [theta.copy()]
        step_count = 0

        for x in xs_iter:
            theta = self.step(theta, x, rng=rng)
            thetas.append(theta.copy())
            step_count += 1
            if n_steps is not None and step_count >= n_steps:
                break

        return np.array(thetas)


class SGLD:
	"""
	Stochastic Gradient Langevin Dynamics.

	Update rule:
		theta_{n+1} = theta_n - lr * H(theta_n, x_{n+1})
					  + sqrt(2 * lr / beta) * xi_{n+1}
	"""

	def __init__(self, H, lr, beta):
		# H: stochastic gradient H(theta, x)
		# lr: step size lambda > 0
		# beta: inverse temperature > 0
		self.H = H
		self.lr = float(lr)
		self.beta = float(beta)

		if self.lr <= 0:
			raise ValueError("lr must be positive")
		if self.beta <= 0:
			raise ValueError("beta must be positive")

	def step(self, theta, x, rng=None):
		if rng is None:
			rng = np.random.default_rng()

		theta = np.asarray(theta, dtype=float)

		noise_scale = np.sqrt(2.0 * self.lr / self.beta)
		xi = rng.normal(0.0, 1.0, size=theta.shape)

		theta_next = theta - self.lr * self.H(theta, x) + noise_scale * xi
		return theta_next

	def simulate(self, theta0, xs, n_steps=None, rng=None):
		# theta0: initial parameter
		# xs: iterable of stochastic samples / mini-batches
		# n_steps: if set, stop after n_steps updates
		if rng is None:
			rng = np.random.default_rng()

		theta = np.asarray(theta0, dtype=float)
		xs_iter = iter(xs)

		thetas = [theta.copy()]
		step_count = 0

		for x in xs_iter:
			theta = self.step(theta, x, rng=rng)
			thetas.append(theta.copy())
			step_count += 1
			if n_steps is not None and step_count >= n_steps:
				break

		return np.array(thetas)


class sTULA:
	"""
	Split Tamed Unadjusted Langevin Algorithm (sTULA).

	Splits the gradient into a strongly convex part (a*theta) and a remainder,
	and applies taming only to the remainder part.

	Update rule:
		f_lambda(theta) = (H(theta, x) - a*theta) / (1 + sqrt(lr) * ||theta||^(2r))
		h_lambda(theta) = a*theta + f_lambda(theta)
		theta_{n+1} = theta_n - lr * h_lambda(theta_n) + sqrt(2*lr/beta) * xi_{n+1}
	"""

	def __init__(self, h, lr, beta, m, r):
		# h: exact gradient h(theta, x)
		# lr: step size lambda > 0
		# beta: inverse temperature > 0
		# m: strong convexity coefficient (equivalent to 'm' in other algorithms) > 0
		# r: taming exponent >= 0
		self.h = h
		self.lr = float(lr)
		self.beta = float(beta)
		self.m = float(m)
		self.r = float(r)

		if self.lr <= 0:
			raise ValueError("lr must be positive")
		if self.beta <= 0:
			raise ValueError("beta must be positive")
		if self.m < 0:
			raise ValueError("m (strong convexity) must be non-negative")
		if self.r < 0:
			raise ValueError("r must be non-negative")

	def _taming_denominator(self, theta):
		"""Taming denominator: 1 + sqrt(lr) * ||theta||^(2r)"""
		theta_norm = np.linalg.norm(theta)
		return 1.0 + np.sqrt(self.lr) * (theta_norm ** (2.0 * self.r))

	def h_lambda(self, theta, x):
		"""
		Split tamed update direction.
		h_lambda(theta) = m*theta + (h(theta,x) - m*theta) / (1 + sqrt(lr)*||theta||^(2r))
		"""
		f = self.h(theta, x) - self.m * theta
		f_lambda = f / self._taming_denominator(theta)
		return self.m * theta + f_lambda

	def step(self, theta, x, rng=None):
		if rng is None:
			rng = np.random.default_rng()

		theta = np.asarray(theta, dtype=float)
		x = np.asarray(x, dtype=float)

		noise_scale = np.sqrt(2.0 * self.lr / self.beta)
		xi = rng.normal(0.0, 1.0, size=theta.shape)

		theta_next = theta - self.lr * self.h_lambda(theta, x) + noise_scale * xi
		return theta_next

	def simulate(self, theta0, xs, n_steps=None, rng=None):
		# theta0: initial parameter
		# xs: iterable of stochastic samples / mini-batches
		# n_steps: if set, stop after n_steps updates
		if rng is None:
			rng = np.random.default_rng()

		theta = np.asarray(theta0, dtype=float)
		xs_iter = iter(xs)

		thetas = [theta.copy()]
		step_count = 0

		for x in xs_iter:
			theta = self.step(theta, x, rng=rng)
			thetas.append(theta.copy())
			step_count += 1
			if n_steps is not None and step_count >= n_steps:
				break

		return np.array(thetas)
