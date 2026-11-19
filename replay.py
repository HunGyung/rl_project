"""Proportional PER: sum/min tree, replacement sampling, normalized IS weights."""
import numpy as np


class PrioritizedReplay:
    def __init__(self, capacity, alpha=0.6, epsilon=1e-6, seed=0):
        if capacity < 1 or not 0 <= alpha <= 1 or epsilon <= 0:
            raise ValueError("Invalid replay settings")
        self.capacity, self.alpha, self.epsilon = capacity, alpha, epsilon
        self.leaves = 1 << (capacity - 1).bit_length()
        self.sums = np.zeros(2 * self.leaves, dtype=np.float64)
        self.mins = np.full(2 * self.leaves, np.inf, dtype=np.float64)
        self.items = [None] * capacity
        self.position = self.size = 0
        self.max_priority = 1.0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def _set(self, indices, priorities):
        # 중복 표본은 최대 TD 오차를 사용해 결정적으로 갱신한다.
        unique, inverse = np.unique(np.asarray(indices, dtype=np.int64), return_inverse=True)
        values = np.zeros(len(unique), dtype=np.float64)
        np.maximum.at(values, inverse, priorities)
        nodes = unique + self.leaves
        self.sums[nodes] = values ** self.alpha
        self.mins[nodes] = self.sums[nodes]
        while np.any(nodes > 1):
            nodes = np.unique(nodes[nodes > 1] // 2)
            self.sums[nodes] = self.sums[2 * nodes] + self.sums[2 * nodes + 1]
            self.mins[nodes] = np.minimum(self.mins[2 * nodes], self.mins[2 * nodes + 1])

    def append(self, item):
        self.items[self.position] = item
        # 새 경험은 지금까지 관측한 최대 우선순위로 삽입한다.
        node = self.position + self.leaves
        self.sums[node] = self.mins[node] = self.max_priority ** self.alpha
        while node > 1:
            node //= 2
            self.sums[node] = self.sums[2 * node] + self.sums[2 * node + 1]
            self.mins[node] = min(self.mins[2 * node], self.mins[2 * node + 1])
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta):
        if not self.size or batch_size < 1 or not 0 <= beta <= 1:
            raise ValueError("Invalid sample request")
        masses = self.rng.random(batch_size) * self.sums[1]
        nodes = np.ones(batch_size, dtype=np.int64)
        while nodes[0] < self.leaves:
            left = nodes * 2
            right = masses >= self.sums[left]
            masses -= np.where(right, self.sums[left], 0)
            nodes = left + right
        indices = nodes - self.leaves
        probabilities = self.sums[nodes] / self.sums[1]
        min_probability = self.mins[1] / self.sums[1]
        # (N P(i))^-beta를 버퍼 전체의 최대 가중치로 정규화한다.
        weights = (probabilities / min_probability) ** (-beta)
        return [self.items[i] for i in indices], indices, weights.astype(np.float32)

    def update(self, indices, td_errors):
        errors = np.asarray(td_errors, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)
        if errors.shape != indices.shape or not np.all(np.isfinite(errors)):
            raise ValueError("Invalid TD errors")
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise ValueError("Invalid replay index")
        priorities = np.abs(errors) + self.epsilon
        self.max_priority = max(self.max_priority, float(priorities.max()))
        self._set(indices, priorities)
