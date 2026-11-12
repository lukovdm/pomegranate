# test_pomdp.py

import itertools

import numpy
import torch
import pytest

from pomegranate.hmm import DenseHMM
from pomegranate.hmm import DensePOMDP
from pomegranate.hmm import LabelledMDP
from pomegranate.hmm import encode_traces
from pomegranate.distributions import Categorical
from pomegranate.distributions import Normal


def _street_crossing(p=0.3):
	"""A small nondeterministic labelled MDP in the spirit of Fig. 6.

	Labels: 0=start, 1=left, 2=right, 3=bump, 4=avoid. Actions: 0=stay,
	1=move. States: 0=s1 (agent and stranger on the same side), 1=s2 (other
	side), 2=hit, 3=ok.
	"""

	k, L, A = 4, 5, 2
	iota = torch.zeros(L, k)
	iota[0, 0] = 1.0

	tau = torch.zeros(A, k, L, k)
	# stay on the same side: the stranger keeps coming (bump) or switches.
	tau[0, 0, 3, 2] = 1 - p
	tau[0, 0, 2, 1] = p
	# move to the other side: the stranger follows with probability p.
	tau[1, 0, 1, 1] = 1 - p
	tau[1, 0, 3, 2] = p
	tau[0, 1, 4, 3] = 1 - p
	tau[0, 1, 1, 0] = p
	tau[1, 1, 2, 0] = 1 - p
	tau[1, 1, 4, 3] = p
	for a in range(A):
		tau[a, 2, 3, 2] = 1.0
		tau[a, 3, 4, 3] = 1.0

	return LabelledMDP(k, starts=iota, edges=tau, random_state=0)


def _brute_force_labelled(model, x):
	iota, tau = torch.exp(model.starts), torch.exp(model.edges)
	T = x.shape[0]
	total, best, best_path = 0.0, -1.0, None
	for path in itertools.product(range(model.n_states), repeat=T):
		p = iota[x[0, 0], path[0]]
		for t in range(1, T):
			p = p * tau[x[t-1, 1], path[t-1], x[t, 0], path[t]]
		total += p
		if p > best:
			best, best_path = p, path
	return torch.log(total), best_path


@pytest.fixture
def X_labelled():
	return torch.tensor([
		[[0, 1], [1, 0], [1, 1], [2, 0], [0, 0]],
		[[2, 0], [1, 1], [0, 1], [2, 0], [1, 0]],
		[[1, 1], [1, 1], [0, 0], [0, 1], [2, 0]],
	])


def test_encode_traces():
	X, labels, actions = encode_traces([
		["init", "up", "grass", "left", "mud"],
		["init", "left", "mud"]
	])

	assert labels == ["init", "grass", "mud"]
	assert actions == ["up", "left"]
	assert X[0].tolist() == [[0, 0], [1, 1], [2, 0]]
	assert X[1].tolist() == [[0, 1], [2, 0]]

	with pytest.raises(ValueError):
		encode_traces([["init", "up"]])


def test_labelled_brute_force(X_labelled):
	model = LabelledMDP(3, n_labels=3, n_actions=2, random_state=0)
	model._initialize(*model._split(X_labelled))

	logp = model.log_probability(X_labelled)
	paths = model.viterbi(X_labelled)
	for i, x in enumerate(X_labelled):
		logp_bf, path_bf = _brute_force_labelled(model, x)
		assert torch.allclose(logp[i], logp_bf.float(), atol=1e-5)
		assert paths[i].tolist() == list(path_bf)

	# Posteriors are distributions and filtered beliefs too.
	assert torch.allclose(model.predict_proba(X_labelled).sum(-1),
		torch.ones(3, 5), atol=1e-5)
	assert torch.allclose(model.belief(X_labelled).sum(-1),
		torch.ones(3, 5), atol=1e-5)


def test_separate_actions(X_labelled):
	model = LabelledMDP(3, n_labels=3, n_actions=2, random_state=0)
	model._initialize(*model._split(X_labelled))

	logp = model.log_probability(X_labelled)
	logp2 = model.log_probability(X_labelled[..., :1],
		actions=X_labelled[:, :-1, 1])
	logp3 = model.log_probability(X_labelled[..., :1],
		actions=X_labelled[:, :, 1])
	assert torch.allclose(logp, logp2)
	assert torch.allclose(logp, logp3)


def test_labelled_em_monotone_and_valid(X_labelled):
	model = LabelledMDP(3, max_iter=50, tol=0.0, random_state=0).fit(
		X_labelled)
	h = numpy.array(model.history_)
	assert numpy.all(numpy.diff(h) > -1e-4)

	iota, tau = torch.exp(model.starts), torch.exp(model.edges)
	assert torch.allclose(iota.sum(), torch.tensor(1.0), atol=1e-5)
	assert torch.allclose(tau.sum(dim=(2, 3)), torch.ones(2, 3), atol=1e-5)


def test_labelled_recovers_model():
	true = _street_crossing()
	X = true.sample(2000, 6)
	X_test = true.sample(500, 6)

	model = LabelledMDP(4, tol=1e-3, random_state=1).fit(X)
	ll_true = true.log_probability(X_test).mean()
	ll_model = model.log_probability(X_test).mean()
	assert abs(ll_true - ll_model) < 0.05


def test_variable_length_and_weights(X_labelled):
	seqs = [X_labelled[0], X_labelled[1, :3], X_labelled[2, :4]]
	m1 = LabelledMDP(2, n_labels=3, n_actions=2, max_iter=5, tol=0.0,
		random_state=0).fit(seqs + [seqs[0]])
	m2 = LabelledMDP(2, n_labels=3, n_actions=2, max_iter=5, tol=0.0,
		random_state=0).fit(seqs, sample_weight=[2.0, 1.0, 1.0])

	assert torch.allclose(m1.edges, m2.edges, atol=1e-4)
	assert torch.allclose(m1.starts, m2.starts, atol=1e-4)


def test_impossible_trace_is_ignored():
	true = _street_crossing()
	X = true.sample(50, 4)
	bad = torch.tensor([[[3, 0], [3, 0], [3, 0], [3, 0]]])
	assert torch.isinf(true.log_probability(bad)).all()

	true.max_iter = 2
	true.fit(torch.cat([X, bad]))
	assert torch.isfinite(true.starts[torch.isfinite(true.starts)]).all()
	assert not torch.isnan(true.edges).any()


def test_single_action_matches_dense_hmm():
	dists = [Categorical([[0.7, 0.2, 0.1]]), Categorical([[0.1, 0.3, 0.6]])]
	edges = [[0.8, 0.2], [0.3, 0.7]]
	starts = [0.6, 0.4]

	hmm = DenseHMM(dists, edges=edges, starts=starts, ends=[1.0, 1.0])
	pomdp = DensePOMDP(dists, edges=[edges], starts=starts)

	X = torch.tensor([[[0], [2], [1], [2], [0]], [[1], [1], [2], [0], [0]]])
	actions = torch.zeros(2, 5, dtype=torch.int64)

	assert torch.allclose(hmm.forward(X), pomdp.forward(X, actions),
		atol=1e-5)
	assert torch.allclose(hmm.predict_proba(X),
		pomdp.predict_proba(X, actions), atol=1e-5)
	assert torch.equal(hmm.viterbi(X), pomdp.viterbi(X, actions))


def test_dense_pomdp_brute_force():
	dists = [Normal([0.0], [1.0], 'diag'), Normal([3.0], [1.0], 'diag')]
	edges = torch.tensor([[[0.9, 0.1], [0.2, 0.8]], [[0.1, 0.9], [0.6, 0.4]]])
	starts = torch.tensor([0.3, 0.7])
	model = DensePOMDP(dists, edges=edges, starts=starts)

	x = torch.tensor([[[0.1, 1], [2.5, 0], [3.1, 1], [-0.4, 0]]])
	e = torch.stack([d.log_probability(x[0, :, :1]) for d in dists], dim=1)

	total = 0.0
	for path in itertools.product(range(2), repeat=4):
		p = torch.log(starts[path[0]]) + e[0, path[0]]
		for t in range(1, 4):
			p = p + torch.log(edges[int(x[0, t-1, 1]), path[t-1], path[t]])
			p = p + e[t, path[t]]
		total += torch.exp(p)

	assert torch.allclose(model.log_probability(x), torch.log(total),
		atol=1e-5)


def test_dense_pomdp_recovers_actions():
	true = DensePOMDP(
		[Normal([0.0], [0.5], 'diag'), Normal([4.0], [0.5], 'diag')],
		edges=[[[0.95, 0.05], [0.05, 0.95]], [[0.05, 0.95], [0.95, 0.05]]],
		starts=[0.5, 0.5], random_state=0)
	X = true.sample(300, 15)

	model = DensePOMDP([Normal(), Normal()], tol=1e-3,
		random_state=0).fit(X)

	# Match states by their means (the labelling is arbitrary).
	order = torch.argsort(torch.stack([d.means[0] for d in
		model.distributions]))
	edges = torch.exp(model.edges)[:, order][:, :, order]
	assert torch.allclose(edges, torch.exp(true.edges), atol=0.06)
	assert torch.allclose(torch.stack([model.distributions[i].means[0] for i
		in order]), torch.tensor([0.0, 4.0]), atol=0.1)


def test_dense_pomdp_categorical_lists():
	true = DensePOMDP(
		[Categorical([[0.8, 0.1, 0.1]]), Categorical([[0.1, 0.1, 0.8]])],
		edges=[[[0.9, 0.1], [0.1, 0.9]], [[0.2, 0.8], [0.8, 0.2]]],
		starts=[1.0, 0.0], random_state=0)
	X = [true.sample(1, l)[0] for l in (3, 5, 5, 7, 8, 3)]

	model = DensePOMDP([Categorical(), Categorical()], max_iter=5,
		random_state=0).fit(X)
	assert torch.isfinite(model.log_probability(torch.stack(X[1:3]))).all()


def test_active_sampling():
	true = _street_crossing()
	X = true.sample(30, 6)
	model = LabelledMDP(4, max_iter=20, random_state=0).fit(X)

	counts = model.expected_action_counts(X)
	assert counts.shape == (4, 2)
	assert torch.allclose(counts.sum(), torch.tensor(30 * 5.0), atol=1e-3)

	sigma = model.scheduler(counts)
	assert torch.allclose(sigma.sum(1), torch.ones(4), atol=1e-6)
	assert torch.all(sigma >= 0)

	# A simulator of the true system to query.
	class System:
		def __init__(self, model):
			self.model = model
			self.rng = numpy.random.RandomState(0)
			self.tau = torch.exp(model.edges).numpy().reshape(2, 4, -1)

		def reset(self):
			self.s = 0
			return 0

		def step(self, a):
			idx = self.rng.choice(self.tau.shape[-1], p=self.tau[a, self.s])
			l, self.s = divmod(idx, 4)
			return l

	system = System(true)
	before = counts.sum().item()
	new = model.active_sample(system.reset, system.step, 6, counts, n=10)
	assert len(new) == 10 and new[0].shape == (6, 2)
	assert counts.sum().item() == pytest.approx(before + 50, abs=1e-3)
	assert torch.isfinite(true.log_probability(torch.stack(new))).all()

	model.fit(list(X) + new)


def test_active_sampling_unexpected_observation():
	# The hypothesis says label 1 never occurs; the system only emits 1.
	starts = torch.tensor([[0.5, 0.5], [0.0, 0.0]])
	edges = torch.zeros(2, 2, 2, 2)
	edges[:, :, 0, :] = 0.5
	model = LabelledMDP(2, starts=starts, edges=edges, random_state=0)

	counts = torch.zeros(2, 2)
	X = model.active_sample(lambda: 1, lambda a: 1, 5, counts, n=3)
	assert len(X) == 3
	assert torch.isfinite(counts).all()
	assert counts.sum().item() == pytest.approx(12.0)
