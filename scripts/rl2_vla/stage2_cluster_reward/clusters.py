"""Episode-balanced, train-only outcome-cluster potentials (not causal labels)."""
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


class ClusterPotential:
    def __init__(self, tasks, metadata):
        self.tasks, self.metadata = tasks, metadata

    def __call__(self, task, context):
        if task not in self.tasks:
            raise ValueError(f'unknown reward task {task}')
        b = self.tasks[task]
        if b.get('disabled'):
            return 0.
        z = ((np.asarray(context) - b['mean']) / b['std']) @ b['components'].T
        k = np.argmin(((b['centers'] - z)**2).sum(axis=1))
        return float(b['scores'][k])

    def save(self, path):
        def encode(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            raise TypeError(type(v).__name__)
        Path(path).write_text(json.dumps(dict(metadata=self.metadata, tasks=self.tasks), default=encode, indent=2))

    @classmethod
    def load(cls, path):
        b = json.loads(Path(path).read_text())
        for t in b['tasks'].values():
            if not t.get('disabled'):
                for key in ('mean', 'std', 'components', 'centers', 'scores'):
                    t[key] = np.asarray(t[key])
        return cls(b['tasks'], b['metadata'])


def fit(episodes, manifest_sha256, k=48, dim=16, seed=42):
    bundles = {}
    for task in sorted({e['task_id'] for e in episodes}):
        eps = [e for e in episodes if e['task_id'] == task]
        if len({e['success'] for e in eps}) < 2:
            bundles[task] = {'disabled': 'one outcome class; potential fixed at zero'}
            continue
        xs = [np.asarray([c['context'] for c in e['chunks']], np.float64) for e in eps]
        x = np.concatenate(xs)
        if len(x) < max(k, dim):
            raise ValueError(f'{task}: fewer than {max(k,dim)} contexts; collect more, do not silently change K')
        w = np.concatenate([np.full(len(a), 1/len(a)) for a in xs])
        w /= w.sum()
        mean = np.sum(x*w[:, None], axis=0)
        std = np.sqrt(np.sum((x-mean)**2*w[:, None], axis=0)).clip(1e-6)
        z = (x-mean)/std
        # Weighted PCA: long episodes cannot dominate the coordinate transform.
        _, _, vt = np.linalg.svd(z*np.sqrt(w[:, None]), full_matrices=False)
        components = vt[:dim]
        projected = z@components.T
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(projected, sample_weight=w)
        counts, support, cursor = [], np.zeros(k), 0
        for a in xs:
            labels = km.labels_[cursor:cursor+len(a)]
            n = np.bincount(labels, minlength=k)
            counts.append(n / len(a))
            support += n > 0
            cursor += len(a)
        counts = np.asarray(counts)
        outcomes = np.asarray([e['success'] for e in eps])
        s, f = counts[outcomes].mean(0), counts[~outcomes].mean(0)
        scores = (s-f)/(s+f+1e-12) * support/(support+5.)
        bundles[task] = dict(mean=mean, std=std, components=components,
                             centers=km.cluster_centers_, scores=scores, support=support,
                             success_occupancy=s, failure_occupancy=f)
    return ClusterPotential(bundles, dict(manifest_sha256=manifest_sha256, split='train',
        k=k, dim=dim, seed=seed, episode_ids=[e['episode_id'] for e in episodes]))
