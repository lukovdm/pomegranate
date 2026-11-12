# pomdp.py
# Partially observable Markov decision processes trained with an
# action-conditioned variant of the Baum-Welch algorithm.

"""Partially observable MDPs learned with Baum-Welch.

This module extends the hidden Markov models in pomegranate to the
controlled setting: the hidden dynamics are driven by a sequence of actions
that is *observed* (it is part of the data) but not modelled. Learning is
done with an action-conditioned version of Baum-Welch, following

	G. Bacci, A. Ingolfsdottir, K. G. Larsen, R. Reynouard. "Active Learning
	of Markov Decision Processes using Baum-Welch algorithm". ICMLA 2021,
	arXiv:2110.03014.

Two models are provided.

	``LabelledMDP``: the model of the paper. A labelled MDP
	M = <S, L, A, iota, {tau_a}> with an initial distribution iota(l, s) over
	a first label and a first state, and a transition function
	tau_a(s)(l', s') that, upon receiving action a in state s, jointly emits
	the next label l' and moves to s'. Labels and actions are categorical.
	This is MDP-BW; with a single action it is MC-BW.

	``DensePOMDP``: the factored POMDP P(s'|s, a) * P(o|s'), where the
	observation model is a set of arbitrary pomegranate distributions (one per
	hidden state), so observations can be continuous, multivariate, etc.

Data format
-----------
A trace is l_1, a_1, l_2, a_2, ..., a_{T-1}, l_T: an observation followed by
the action taken after it. A trace is stored as a (T, d+1) tensor where row
t is ``[observation_t..., action_t]``, i.e. the *last column holds the
action that was taken after the observation in that row*. The action in the
last row is ignored (there is no next observation), use any valid integer
such as 0. Alternatively, pass observations as X and the actions separately
with ``actions=``, of shape (n, T) or (n, T-1).

Like the HMMs, X can be a 3D tensor (n, T, d+1) of equal-length traces, a
list of such tensors, or a list of 2D (T, d+1) tensors of varying lengths.
``encode_traces`` converts flat traces such as
``['init', 'a', 'x', 'b', 'y']`` to this format.
"""

import time
import numpy
import torch

from .._utils import _cast_as_tensor
from .._utils import _cast_as_parameter
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import partition_sequences

from ..distributions._distribution import Distribution
from ..kmeans import KMeans


NEGINF = float("-inf")


def encode_traces(traces, labels=None, actions=None):
	"""Convert flat traces l_1, a_1, l_2, ..., l_T to integer tensors.

	Parameters
	----------
	traces: list of sequences
		Each trace alternates observation labels and actions, starting and
		ending with a label, e.g. ``['init', 'up', 'grass', 'left', 'mud']``.

	labels: list or None, optional
		The label alphabet, defining the integer encoding. If None, labels are
		encoded in order of first appearance.

	actions: list or None, optional
		The action alphabet. If None, encoded in order of first appearance.


	Returns
	-------
	X: list of torch.Tensor, shape=(T, 2)
		Row t is [label index, action index]; the last action is 0.

	labels: list
		The label alphabet; ``labels[i]`` is encoded as i.

	actions: list
		The action alphabet; ``actions[i]`` is encoded as i.
	"""

	label_idx = {} if labels is None else {l: i for i, l in enumerate(labels)}
	action_idx = {} if actions is None else {a: i for i, a in
		enumerate(actions)}
	fixed_labels, fixed_actions = labels is not None, actions is not None

	def _code(table, key, fixed, name):
		if key not in table:
			if fixed:
				raise ValueError("Unknown {}: {}".format(name, key))
			table[key] = len(table)
		return table[key]

	X = []
	for trace in traces:
		trace = list(trace)
		if len(trace) % 2 != 1:
			raise ValueError("A trace must alternate labels and actions and " +
				"start and end with a label.")

		rows = []
		for t in range(0, len(trace), 2):
			l = _code(label_idx, trace[t], fixed_labels, "label")
			a = 0 if t+1 >= len(trace) else _code(action_idx, trace[t+1],
				fixed_actions, "action")
			rows.append([l, a])

		X.append(torch.tensor(rows, dtype=torch.int64))

	labels = sorted(label_idx, key=label_idx.get)
	actions = sorted(action_idx, key=action_idx.get)
	return X, labels, actions


def _random_simplex(random_state, shape, dtype):
	"""Uniformly random distributions over the last axis (Dirichlet(1))."""

	p = random_state.dirichlet(numpy.ones(shape[-1]), size=shape[:-1])
	return torch.tensor(p, dtype=dtype)


def _merge_actions(X, actions):
	"""Append the actions as the last column of the observations."""

	def _merge(x, a):
		x, a = _cast_as_tensor(x), _cast_as_tensor(a)
		if a.shape[-1] == x.shape[-2] - 1:
			pad = torch.zeros(*a.shape[:-1], 1, dtype=a.dtype, device=a.device)
			a = torch.cat([a, pad], dim=-1)

		if a.shape != x.shape[:-1]:
			raise ValueError("actions must have shape (..., T) or (..., T-1) " +
				"where X has shape (..., T, d).")

		return torch.cat([x, a.unsqueeze(-1).to(x.dtype)], dim=-1)

	if isinstance(X, (list, tuple)):
		return [_merge(x, a) for x, a in zip(X, actions)]
	return _merge(X, actions)


class _BasePOMDP(Distribution):
	"""Shared inference and learning code for action-driven hidden models.

	Subclasses describe the model through a "context" computed once per batch
	of sequences, and provide the log initial weights and the log step
	matrices:

		_prepare(obs) -> ctx
		_log_init(ctx) -> (n, k): log P(o_1, s_1)
		_log_step(ctx, i, a) -> (n, k, k): log P(o_i, s_i = s' | s_{i-1} = s, a)

	The forward/backward recursions are then those of eqs. (1)-(2) in Bacci
	et al., which reduce to the usual HMM recursions for factored models.
	"""

	def __init__(self, n_actions=None, max_iter=1000, tol=0.1,
		inertia=0.0, frozen=False, check_data=True, random_state=None,
		verbose=False):
		super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)

		if not isinstance(random_state, numpy.random.RandomState):
			self.random_state = numpy.random.RandomState(random_state)
		else:
			self.random_state = random_state

		self.n_actions = n_actions
		self.max_iter = _check_parameter(max_iter, "max_iter", min_value=1,
			ndim=0, dtypes=(int, torch.int32, torch.int64))
		self.tol = _check_parameter(tol, "tol", min_value=0., ndim=0)
		self.verbose = verbose

	# ------------------------------------------------------------------
	# Data handling
	# ------------------------------------------------------------------

	def _split(self, X, actions=None):
		"""Return (obs, actions) with obs (n, T, d) and actions (n, T)."""

		if actions is not None:
			X = _merge_actions(X, actions)

		X = _check_parameter(_cast_as_tensor(X), "X", ndim=3,
			check_parameter=self.check_data)
		if X.shape[-1] < 2:
			raise ValueError("X must contain at least one observation column "
				+ "and the action column.")

		obs, a = X[..., :-1], X[..., -1]
		if obs.is_floating_point() and self._integer_observations:
			obs = obs.long()

		a = a.long()
		if self.check_data and self.n_actions is not None and X.shape[1] > 1:
			_check_parameter(a[:, :-1], "actions", min_value=0,
				max_value=self.n_actions-1)

		return obs.to(self.device), a.to(self.device)

	def _partition(self, X, actions=None, sample_weight=None):
		if actions is not None:
			X = _merge_actions(X, actions)

		X, sample_weight, _ = partition_sequences(X,
			sample_weight=sample_weight)
		return X, sample_weight

	def _infer_n_actions(self, X):
		if self.n_actions is None:
			self.n_actions = int(max(int(x[..., -1].max()) for x in X)) + 1

	@property
	def _integer_observations(self):
		return False

	# ------------------------------------------------------------------
	# Inference
	# ------------------------------------------------------------------

	def _forward(self, ctx, a):
		f0 = self._log_init(ctx)
		l = a.shape[1]

		f = [f0]
		for i in range(1, l):
			m = self._log_step(ctx, i, a[:, i-1])
			f.append(torch.logsumexp(f[-1].unsqueeze(-1) + m, dim=1))

		return torch.stack(f, dim=1)

	def _backward(self, ctx, a):
		n, l = a.shape
		b = [torch.zeros(n, self.n_states, dtype=self.dtype,
			device=self.device)]

		for i in range(l-1, 0, -1):
			m = self._log_step(ctx, i, a[:, i-1])
			b.append(torch.logsumexp(m + b[-1].unsqueeze(1), dim=2))

		return torch.stack(b[::-1], dim=1)

	def forward(self, X, actions=None):
		"""Run the forward algorithm, eq. (1) of Bacci et al.

		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, T, d+1)
			Equal-length traces whose last column contains the actions (or
			shape (-1, T, d) when `actions` is given).

		actions: torch.Tensor or None, shape=(-1, T) or (-1, T-1), optional
			The actions, if not included in X.


		Returns
		-------
		f: torch.Tensor, shape=(-1, T, self.n_states)
			f[r, t, s] = log P(o_1..o_t, s_t = s | a_1..a_{t-1}).
		"""

		obs, a = self._split(X, actions)
		return self._forward(self._prepare(obs), a)

	def backward(self, X, actions=None):
		"""Run the backward algorithm, eq. (2) of Bacci et al.

		Returns
		-------
		b: torch.Tensor, shape=(-1, T, self.n_states)
			b[r, t, s] = log P(o_{t+1}..o_T | s_t = s, a_t..a_{T-1}).
		"""

		obs, a = self._split(X, actions)
		return self._backward(self._prepare(obs), a)

	def log_probability(self, X, actions=None):
		"""The log likelihood of each trace given its actions.

		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			log P(o_1..o_T | a_1..a_{T-1}) for each trace.
		"""

		return torch.logsumexp(self.forward(X, actions)[:, -1], dim=-1)

	def probability(self, X, actions=None):
		return torch.exp(self.log_probability(X, actions))

	def belief(self, X, actions=None):
		"""The filtered belief state P(s_t | o_1..o_t, a_1..a_{t-1}).

		Returns
		-------
		belief: torch.Tensor, shape=(-1, T, self.n_states)
		"""

		f = self.forward(X, actions)
		return torch.exp(f - torch.logsumexp(f, dim=-1, keepdim=True))

	def predict_log_proba(self, X, actions=None):
		"""The smoothed log posterior log P(s_t | o_1..o_T, a), i.e. gamma.

		Returns
		-------
		r: torch.Tensor, shape=(-1, T, self.n_states)
		"""

		obs, a = self._split(X, actions)
		ctx = self._prepare(obs)
		r = self._forward(ctx, a) + self._backward(ctx, a)
		return r - torch.logsumexp(r, dim=-1, keepdim=True)

	def predict_proba(self, X, actions=None):
		return torch.exp(self.predict_log_proba(X, actions))

	def predict(self, X, actions=None):
		"""The most likely hidden state at each step (posterior decoding)."""

		return torch.argmax(self.predict_log_proba(X, actions), dim=-1)

	def viterbi(self, X, actions=None):
		"""The single most likely hidden state path for each trace.

		Returns
		-------
		path: torch.Tensor, shape=(-1, T)
		"""

		obs, a = self._split(X, actions)
		ctx = self._prepare(obs)
		n, l = a.shape

		v = self._log_init(ctx)
		traceback = []
		for i in range(1, l):
			z = v.unsqueeze(-1) + self._log_step(ctx, i, a[:, i-1])
			v, idx = torch.max(z, dim=1)
			traceback.append(idx)

		path = [torch.argmax(v, dim=-1)]
		for idx in traceback[::-1]:
			path.append(idx[torch.arange(n), path[-1]])

		return torch.stack(path[::-1], dim=1)

	# ------------------------------------------------------------------
	# Learning
	# ------------------------------------------------------------------

	def summarize(self, X, actions=None, sample_weight=None):
		"""Accumulate expected sufficient statistics (the E step).

		Computes gamma and xi (eq. (3) of Bacci et al.) with the
		forward-backward algorithm and adds them to the stored statistics.
		Traces that are impossible under the current model (likelihood zero)
		contribute nothing.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor, shape=(-1, T, d+1)
			Equal-length traces with the actions in the last column.

		actions: torch.Tensor or None, optional
			The actions, if not included in X.

		sample_weight: torch.Tensor or None, shape=(-1,), optional
			A weight per trace. Default is ones.


		Returns
		-------
		logp: torch.Tensor, shape=(-1,)
			The log likelihood of each trace under the model before the update.
		"""

		obs, a = self._split(X, actions)
		if not self._initialized:
			self._initialize(obs, a)

		n, l = a.shape
		if sample_weight is None:
			w = torch.ones(n, dtype=self.dtype, device=self.device)
		else:
			w = _check_parameter(_cast_as_tensor(sample_weight,
				dtype=self.dtype), "sample_weight", min_value=0., ndim=1,
				shape=(n,), check_parameter=self.check_data).to(self.device)

		ctx = self._prepare(obs)
		f = self._forward(ctx, a)
		b = self._backward(ctx, a)
		logp = torch.logsumexp(f[:, -1], dim=-1)

		# Impossible traces get weight zero rather than producing NaNs.
		valid = torch.isfinite(logp)
		w = torch.where(valid, w, torch.zeros_like(w))
		logp_ = torch.where(valid, logp, torch.zeros_like(logp))

		gamma = torch.exp(f + b - logp_[:, None, None]) * w[:, None, None]
		gamma = torch.nan_to_num(gamma)

		self._summarize_emissions(obs, ctx, gamma)
		if self.frozen:
			return logp

		self._summarize_init(ctx, gamma[:, 0])
		for i in range(1, l):
			ai = a[:, i-1]
			xi = (f[:, i-1].unsqueeze(-1) + self._log_step(ctx, i, ai)
				+ b[:, i].unsqueeze(1) - logp_[:, None, None])
			xi = torch.nan_to_num(torch.exp(xi) * w[:, None, None])
			self._summarize_step(ctx, i, ai, xi)

		return logp

	def fit(self, X, actions=None, sample_weight=None):
		"""Fit the model with Baum-Welch (MDP-BW, Fig. 1 of Bacci et al.).

		Iterates the E step (`summarize`) and M step (`from_summaries`) until
		the improvement in total log likelihood is below `tol` or `max_iter`
		iterations have been performed.


		Parameters
		----------
		X: list, numpy.ndarray, torch.Tensor
			Traces with the action in the last column: either a (n, T, d+1)
			tensor, a list of such tensors, or a list of (T, d+1) tensors of
			varying lengths.

		actions: optional
			Actions in the same layout as X (without the last dimension), if
			they are not included in X.

		sample_weight: optional
			One weight per trace, in the same layout as X.


		Returns
		-------
		self
		"""

		X, sample_weight = self._partition(X, actions, sample_weight)
		self._infer_n_actions(X)

		if not self._initialized:
			obs, a = zip(*[self._split(x) for x in X])
			self._initialize(obs, a, sample_weight)

		last_logp = None
		self.history_ = []
		for i in range(self.max_iter+1):
			start_time = time.time()

			logp = 0
			for j, X_ in enumerate(X):
				w_ = None if sample_weight is None else sample_weight[j]
				logp += self.summarize(X_, sample_weight=w_).sum()

			self.history_.append(float(logp))
			if last_logp is not None:
				improvement = logp - last_logp
				if self.verbose:
					print("[{}] Improvement: {}, Time: {:4.4}s".format(i,
						improvement, time.time() - start_time))

				if improvement < self.tol:
					break

			if i == self.max_iter:
				break

			last_logp = logp
			self.from_summaries()

		self._reset_cache()
		return self

	# ------------------------------------------------------------------
	# Active learning (Section IV of Bacci et al.)
	# ------------------------------------------------------------------

	def expected_action_counts(self, X, actions=None):
		"""Expected number of times each action was chosen from each state.

		Eq. (4) of Bacci et al.: m_sa = sum_r sum_t 1_a(a_t) gamma(s, t).


		Returns
		-------
		m: torch.Tensor, shape=(self.n_states, self.n_actions)
		"""

		X, _ = self._partition(X, actions)
		m = torch.zeros(self.n_states, self.n_actions, dtype=self.dtype,
			device=self.device)

		for X_ in X:
			_, a = self._split(X_)
			gamma = self.predict_proba(X_)[:, :-1]
			onehot = torch.nn.functional.one_hot(a[:, :-1], self.n_actions)
			m += torch.einsum("nts,nta->sa", gamma, onehot.to(gamma.dtype))

		return m

	def scheduler(self, counts):
		"""The memoryless scheduler sigma_M of eq. (5) of Bacci et al.

		sigma_M(s)(a) is proportional to 1 - m_sa / sum_a' m_sa', favouring
		actions that were rarely chosen in state s. The paper's expression is
		normalized here so it is a distribution for any number of actions.
		States with no counts (and models with one action) are uniform.


		Returns
		-------
		sigma: torch.Tensor, shape=(self.n_states, self.n_actions)
		"""

		counts = _cast_as_tensor(counts, dtype=self.dtype).to(self.device)
		total = counts.sum(dim=1, keepdim=True)
		sigma = 1 - counts / torch.where(total > 0, total, 1)
		sigma = torch.where(total > 0, sigma, torch.ones_like(sigma))
		norm = sigma.sum(dim=1, keepdim=True)
		uniform = torch.ones_like(sigma) / self.n_actions
		return torch.where(norm > 0, sigma / torch.where(norm > 0, norm, 1),
			uniform)

	def active_sample(self, reset, step, length, counts, n=1):
		"""Sample traces from a system with the active strategy of Fig. 5.

		Actions are drawn from the observation-based scheduler of eq. (6),
		sum_s belief(s) sigma_M(s), and the counts m_sa are updated online with
		the current belief, as in lines 6-10 of ActiveSampling.


		Parameters
		----------
		reset: callable
			``reset()`` restarts the system under learning and returns its
			first observation (a scalar, or a sequence of length d).

		step: callable
			``step(action)`` applies the integer action and returns the next
			observation.

		length: int or callable
			The length T of each trace, or a function ``length(random_state)``
			returning one (e.g. shifted geometric lengths).

		counts: torch.Tensor, shape=(self.n_states, self.n_actions)
			The action counts M, e.g. from `expected_action_counts` on the
			current training set. Updated in place.

		n: int, optional
			The number of traces to sample. Default is 1.


		Returns
		-------
		X: list of torch.Tensor, shape=(T, d+1)
			The new traces, in the format accepted by `fit`.
		"""

		traces = []
		for _ in range(n):
			T = length(self.random_state) if callable(length) else length

			o = _cast_as_tensor(reset()).reshape(1, 1, -1)
			alpha = self._log_init(self._prepare(o.to(self.device)))[0]
			if not torch.isfinite(alpha).any():
				alpha = torch.zeros_like(alpha)

			rows = []

			for t in range(T-1):
				sigma = self.scheduler(counts)
				belief = torch.exp(alpha - torch.logsumexp(alpha, dim=0))
				p = (belief @ sigma).double().cpu().numpy()
				a = int(self.random_state.choice(self.n_actions, p=p / p.sum()))

				counts[:, a] += belief
				rows.append(torch.cat([o.reshape(-1).double(),
					torch.tensor([a], dtype=torch.float64)]))

				o = _cast_as_tensor(step(a)).reshape(1, 1, -1)
				m = self._log_step(self._prepare(o.to(self.device)), 0,
					torch.tensor([a], device=self.device))[0]
				alpha = torch.logsumexp(alpha.unsqueeze(-1) + m, dim=0)

				# The hypothesis cannot produce what the system just did (see
				# Sec. IV-A); continue from a uniform belief.
				if not torch.isfinite(alpha).any():
					alpha = torch.zeros_like(alpha)

			rows.append(torch.cat([o.reshape(-1).double(),
				torch.tensor([0], dtype=torch.float64)]))

			x = torch.stack(rows)
			if self._integer_observations:
				x = x.long()
			traces.append(x)

		return traces

	def _sample_action(self, policy, history):
		if policy is None:
			return int(self.random_state.randint(self.n_actions))
		if callable(policy):
			return int(policy(history))
		p = numpy.asarray(policy, dtype=numpy.float64)
		return int(self.random_state.choice(self.n_actions, p=p / p.sum()))

	@staticmethod
	def _normalize(old, counts, dim):
		"""log(counts / sum) over `dim`, keeping `old` where no counts.

		Following Remark 3.1 of Bacci et al., distributions of states that
		were never visited are left unchanged.
		"""

		total = counts.sum(dim=dim, keepdim=True)
		new = torch.log(counts / torch.where(total > 0, total, 1))
		return torch.where(total > 0, new, old)


class LabelledMDP(_BasePOMDP):
	"""A labelled MDP / POMDP with joint label-state transitions (MDP-BW).

	This is the model learned in Bacci et al. (2021). A labelled MDP is a
	tuple <S, L, A, iota, {tau_a}> where iota(l, s) is the probability of
	initially emitting label l and being in state s, and tau_a(s)(l', s') is
	the probability that, when action a is received in state s, the model
	emits label l' and moves to state s'. Because labels are emitted jointly
	with the transition, the observation can depend on the source state, the
	action and the target state. With one action it is a labelled Markov
	chain (MC-BW). Labels and actions are integers.

	Learning is done with the Baum-Welch update of Section III:

		iota(l, s)        = sum_r 1_{l_1^r}(l) gamma_r(s, 1) / R
		tau_a(s)(l, s')   = sum_r sum_t xi_r^a(s, t)(l, s')
		                    / sum_r sum_t 1_a(a_t^r) gamma_r(s, t)


	Parameters
	----------
	n_states: int
		The number of hidden states |S|.

	n_labels: int or None, optional
		The number of labels |L|. Inferred from the data if None.

	n_actions: int or None, optional
		The number of actions |A|. Inferred from the data if None.

	starts: array-like or None, shape=(n_labels, n_states), optional
		The initial distribution iota (sums to one). Random if None.

	edges: array-like or None, shape=(n_actions, n_states, n_labels, n_states)
		The transition function tau; edges[a, s] sums to one. Random if None.

	max_iter: int, optional
		The maximum number of Baum-Welch iterations. Default is 1000.

	tol: float, optional
		Stop when the total log likelihood improves less than this (epsilon
		in Fig. 1). Default is 0.1.

	inertia: float, [0, 1], optional
		The proportion of the old parameters kept at each update.

	frozen: bool, optional
		Whether the parameters are frozen. Default is False.

	random_state: int, numpy.random.RandomState or None, optional
		Controls the random initial hypothesis and sampling.

	verbose: bool, optional
		Whether to print the improvement at each iteration.
	"""

	def __init__(self, n_states, n_labels=None, n_actions=None, starts=None,
		edges=None, max_iter=1000, tol=0.1, inertia=0.0, frozen=False,
		check_data=True, random_state=None, verbose=False):
		super().__init__(n_actions=n_actions, max_iter=max_iter, tol=tol,
			inertia=inertia, frozen=frozen, check_data=check_data,
			random_state=random_state, verbose=verbose)

		self.name = "LabelledMDP"
		self.n_states = n_states
		self.n_labels = n_labels
		self.starts = None
		self.edges = None

		if starts is not None:
			starts = _check_parameter(_cast_as_tensor(starts), "starts",
				ndim=2, min_value=0., max_value=1.)
			self.n_labels = starts.shape[0]
			self.starts = _cast_as_parameter(torch.log(starts))

		if edges is not None:
			edges = _check_parameter(_cast_as_tensor(edges), "edges", ndim=4,
				min_value=0., max_value=1.)
			self.n_actions, _, self.n_labels, _ = edges.shape
			self.edges = _cast_as_parameter(torch.log(edges))

		self._initialized = self.starts is not None and self.edges is not None
		self._reset_cache()

	@property
	def _integer_observations(self):
		return True

	def _initialize(self, obs, a, sample_weight=None):
		if isinstance(obs, torch.Tensor):
			obs, a = [obs], [a]

		if self.n_labels is None:
			self.n_labels = int(max(int(o.max()) for o in obs)) + 1
		if self.n_actions is None:
			self.n_actions = int(max(int(a_[:, :-1].max()) if a_.shape[1] > 1
				else 0 for a_ in a)) + 1

		dtype = self._device.dtype
		k, L, A = self.n_states, self.n_labels, self.n_actions

		if self.starts is None:
			starts = _random_simplex(self.random_state, (L*k,), dtype)
			self.starts = _cast_as_parameter(torch.log(starts.reshape(L, k)))

		if self.edges is None:
			edges = _random_simplex(self.random_state, (A, k, L*k), dtype)
			self.edges = _cast_as_parameter(torch.log(edges.reshape(A, k, L,
				k)))

		self._initialized = True
		self._reset_cache()

	def _reset_cache(self):
		if not self._initialized:
			return

		k, L, A = self.n_states, self.n_labels, self.n_actions
		self.register_buffer("_xw_starts_sum", torch.zeros(L, k,
			dtype=self.dtype, device=self.device))
		self.register_buffer("_xw_sum", torch.zeros(A*L, k, k,
			dtype=self.dtype, device=self.device))

	def _prepare(self, obs):
		labels = obs[..., 0].long()
		if self.check_data:
			_check_parameter(labels, "labels", min_value=0,
				max_value=self.n_labels-1)
		return labels

	def _log_init(self, labels):
		return self.starts[labels[:, 0]]

	def _log_step(self, labels, i, a):
		return self.edges[a, :, labels[:, i], :]

	def _summarize_emissions(self, obs, labels, gamma):
		pass

	def _summarize_init(self, labels, gamma0):
		self._xw_starts_sum.index_add_(0, labels[:, 0], gamma0)

	def _summarize_step(self, labels, i, a, xi):
		self._xw_sum.index_add_(0, a * self.n_labels + labels[:, i], xi)

	def from_summaries(self):
		"""Apply the MDP-BW update (the M step) from the stored statistics."""

		if self.frozen:
			return

		k, L, A = self.n_states, self.n_labels, self.n_actions
		starts = self._normalize(self.starts.reshape(1, -1),
			self._xw_starts_sum.reshape(1, -1), dim=1).reshape(L, k)

		counts = self._xw_sum.reshape(A, L, k, k).permute(0, 2, 1, 3)
		edges = self._normalize(self.edges.reshape(A, k, L*k),
			counts.reshape(A, k, L*k), dim=2).reshape(A, k, L, k)

		_update_parameter(self.starts, starts, inertia=self.inertia)
		_update_parameter(self.edges, edges, inertia=self.inertia)
		self._reset_cache()

	def sample(self, n, length, policy=None, return_states=False):
		"""Sample traces by running the model under a scheduler.

		Parameters
		----------
		n: int
			The number of traces.

		length: int
			The number of labels T in each trace.

		policy: None, array-like of shape (n_actions,), or callable
			The scheduler: uniform if None, fixed action probabilities, or a
			function of the trace so far (a list of (label, action) pairs
			followed by the current label) returning an action.

		return_states: bool, optional
			Whether to also return the hidden state paths.


		Returns
		-------
		X: torch.Tensor, shape=(n, length, 2)
			Rows are [label, action]; the final action is 0.
		"""

		k, L = self.n_states, self.n_labels
		starts = torch.exp(self.starts).reshape(-1).double().cpu().numpy()
		edges = torch.exp(self.edges).reshape(self.n_actions, k, L*k)
		edges = edges.double().cpu().numpy()

		X = numpy.zeros((n, length, 2), dtype=numpy.int64)
		S = numpy.zeros((n, length), dtype=numpy.int64)
		for r in range(n):
			idx = self.random_state.choice(L*k, p=starts / starts.sum())
			X[r, 0, 0], S[r, 0] = divmod(idx, k)

			for t in range(1, length):
				history = [tuple(h) for h in X[r, :t-1].tolist()] + \
					[X[r, t-1, 0]]
				a = self._sample_action(policy, history)
				p = edges[a, S[r, t-1]]
				idx = self.random_state.choice(L*k, p=p / p.sum())
				X[r, t-1, 1] = a
				X[r, t, 0], S[r, t] = divmod(idx, k)

		X, S = torch.from_numpy(X), torch.from_numpy(S)
		return (X, S) if return_states else X


class DensePOMDP(_BasePOMDP):
	"""A POMDP with action-dependent transitions and state emissions.

	The model is the factored POMDP

		P(s_1) P(o_1 | s_1) prod_t P(s_{t+1} | s_t, a_t) P(o_{t+1} | s_{t+1})

	with one dense transition matrix per action and one pomegranate
	distribution per hidden state for the observations, so observations can
	be of any type that pomegranate supports (Categorical, Normal, ...,
	IndependentComponents for multivariate observations). With one action it
	is a DenseHMM without end probabilities. Learning uses the
	action-conditioned Baum-Welch update: transition counts xi are summed
	separately for each action, and emission distributions are updated from
	the posteriors gamma exactly as in an HMM.


	Parameters
	----------
	distributions: list
		One emission distribution per hidden state. They do not need to be
		initialized; uninitialized distributions are initialized with k-means.

	n_actions: int or None, optional
		The number of actions. Inferred from the data if None.

	edges: array-like or None, shape=(n_actions, k, k), optional
		edges[a, s, s'] = P(s' | s, a). Random if None.

	starts: array-like or None, shape=(k,), optional
		The initial state distribution. Uniform if None.

	init: str, optional
		The k-means initialization for the emission distributions.

	max_iter, tol, inertia, frozen, check_data, random_state, verbose:
		As in `LabelledMDP`.
	"""

	def __init__(self, distributions, n_actions=None, edges=None, starts=None,
		init='random', max_iter=1000, tol=0.1, inertia=0.0, frozen=False,
		check_data=True, random_state=None, verbose=False):
		super().__init__(n_actions=n_actions, max_iter=max_iter, tol=tol,
			inertia=inertia, frozen=frozen, check_data=check_data,
			random_state=random_state, verbose=verbose)

		self.name = "DensePOMDP"
		self.distributions = torch.nn.ModuleList(distributions)
		self.n_states = len(distributions)
		self.init = init
		self.starts = None
		self.edges = None
		k = self.n_states

		if starts is not None:
			starts = _check_parameter(_cast_as_tensor(starts), "starts",
				ndim=1, shape=(k,), min_value=0., max_value=1., value_sum=1.0)
			self.starts = _cast_as_parameter(torch.log(starts))

		if edges is not None:
			edges = _check_parameter(_cast_as_tensor(edges), "edges", ndim=3,
				shape=(-1, k, k), min_value=0., max_value=1.)
			self.n_actions = edges.shape[0]
			self.edges = _cast_as_parameter(torch.log(edges))

		self._initialized = (self.starts is not None and self.edges is not None
			and all(d._initialized for d in self.distributions))
		self._reset_cache()

	@property
	def k(self):
		return self.n_states

	def freeze(self):
		super().freeze()
		for d in self.distributions:
			d.freeze()
		return self

	def unfreeze(self):
		super().unfreeze()
		for d in self.distributions:
			d.unfreeze()
		return self

	def _initialize(self, obs, a, sample_weight=None):
		if isinstance(obs, torch.Tensor):
			obs, a = [obs], [a]
			if sample_weight is not None:
				sample_weight = [sample_weight]

		if self.n_actions is None:
			self.n_actions = int(max(int(a_[:, :-1].max()) if a_.shape[1] > 1
				else 0 for a_ in a)) + 1

		k, A = self.n_states, self.n_actions
		dtype = self._device.dtype

		if self.starts is None:
			self.starts = _cast_as_parameter(torch.log(torch.ones(k,
				dtype=dtype) / k))

		if self.edges is None:
			edges = _random_simplex(self.random_state, (A, k, k), dtype)
			self.edges = _cast_as_parameter(torch.log(edges))

		if not all(d._initialized for d in self.distributions):
			X = torch.cat([o.reshape(-1, o.shape[-1]) for o in obs], dim=0)
			if sample_weight is None:
				w = torch.ones(X.shape[0], 1, dtype=dtype)
			else:
				w = torch.cat([_cast_as_tensor(w_).reshape(-1, 1).expand(-1,
					o.shape[1]).reshape(-1, 1) for o, w_ in zip(obs,
					sample_weight)], dim=0).to(dtype)

			model = KMeans(k, init=self.init, max_iter=3,
				random_state=self.random_state)
			y_hat = model.fit_predict(X.to(dtype), sample_weight=w)

			# Fit each distribution on all of the data, weighted mostly towards
			# its own cluster. This keeps empty clusters from failing and lets
			# every categorical distribution see every key.
			for i, d in enumerate(self.distributions):
				if not d._initialized:
					w_i = w * ((y_hat == i).unsqueeze(1).to(dtype) + 1e-2)
					d.fit(X, sample_weight=w_i)

		self._initialized = True
		self._reset_cache()

	def _reset_cache(self):
		if not self._initialized:
			return

		for d in self.distributions:
			d._reset_cache()

		k, A = self.n_states, self.n_actions
		self.register_buffer("_xw_starts_sum", torch.zeros(k,
			dtype=self.dtype, device=self.device))
		self.register_buffer("_xw_sum", torch.zeros(A, k, k,
			dtype=self.dtype, device=self.device))

	def _prepare(self, obs):
		"""Emission log probabilities, shape (n, T, k)."""

		n, l, d = obs.shape
		X = obs.reshape(n*l, d)
		e = torch.empty(n*l, self.n_states, dtype=self.dtype,
			device=self.device)

		for i, dist in enumerate(self.distributions):
			logp = dist.log_probability(X)
			if isinstance(logp, torch.masked.MaskedTensor):
				logp = logp._masked_data
			e[:, i] = logp

		return e.reshape(n, l, self.n_states)

	def _log_init(self, e):
		return self.starts + e[:, 0]

	def _log_step(self, e, i, a):
		return self.edges[a] + e[:, i].unsqueeze(1)

	def _summarize_emissions(self, obs, e, gamma):
		X = obs.reshape(-1, obs.shape[-1])
		for i, d in enumerate(self.distributions):
			d.summarize(X, sample_weight=gamma[:, :, i].reshape(-1, 1))

	def _summarize_init(self, e, gamma0):
		self._xw_starts_sum += gamma0.sum(dim=0)

	def _summarize_step(self, e, i, a, xi):
		self._xw_sum.index_add_(0, a, xi)

	def from_summaries(self):
		"""Update transitions per action and the emission distributions."""

		for d in self.distributions:
			d.from_summaries()

		if self.frozen:
			return

		starts = self._normalize(self.starts, self._xw_starts_sum, dim=0)
		edges = self._normalize(self.edges, self._xw_sum, dim=2)

		_update_parameter(self.starts, starts, inertia=self.inertia)
		_update_parameter(self.edges, edges, inertia=self.inertia)
		self._reset_cache()

	def sample(self, n, length, policy=None, return_states=False):
		"""Sample traces by running the model under a scheduler.

		Parameters
		----------
		n: int
			The number of traces.

		length: int
			The number of observations T in each trace.

		policy: None, array-like of shape (n_actions,), or callable
			The scheduler: uniform if None, fixed action probabilities, or a
			function of the trace so far (a list of observation tensors and
			actions) returning an action.

		return_states: bool, optional
			Whether to also return the hidden state paths.


		Returns
		-------
		X: torch.Tensor, shape=(n, length, d+1)
			Observations with the action taken after each in the last column.
		"""

		starts = torch.exp(self.starts).double().cpu().numpy()
		edges = torch.exp(self.edges).double().cpu().numpy()

		X, S = [], numpy.zeros((n, length), dtype=numpy.int64)
		for r in range(n):
			s = self.random_state.choice(self.n_states, p=starts / starts.sum())
			S[r, 0] = s
			obs = [self.distributions[s].sample(1).reshape(-1)]
			acts, history = [], [obs[0]]

			for t in range(1, length):
				a = self._sample_action(policy, history)
				p = edges[a, s]
				s = self.random_state.choice(self.n_states, p=p / p.sum())
				S[r, t] = s

				acts.append(a)
				obs.append(self.distributions[s].sample(1).reshape(-1))
				history += [a, obs[-1]]

			obs = torch.stack(obs)
			acts = torch.tensor(acts + [0], dtype=obs.dtype).reshape(-1, 1)
			X.append(torch.cat([obs, acts], dim=1))

		X, S = torch.stack(X), torch.from_numpy(S)
		return (X, S) if return_states else X
