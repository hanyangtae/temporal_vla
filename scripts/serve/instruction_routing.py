"""Explicit same-noise routing for instruction setM ablations."""
from __future__ import annotations


def validate_routing(spec, phases):
    if spec is None:
        return None
    if spec.get('version') != 'instruction_setm_routing_v1':
        raise ValueError('unknown instruction routing version')
    if spec.get('mode') not in ('instruction_only', 'cluster_instruction_fallback'):
        raise ValueError('unknown instruction routing mode')
    if 'instruction' not in phases:
        raise ValueError('instruction operator missing')
    if spec['mode'] == 'instruction_only' and set(phases) != {'instruction'}:
        raise ValueError('instruction-only artifact contains cluster operators')
    if not set(phases) <= {'instruction', *(f'c{i}' for i in range(8))}:
        raise ValueError('unexpected routing phase')
    return spec


def resolve_routing(spec, cluster, registered):
    """Return operator key and fallback reason; never requests a new seed."""
    if spec is None:
        return cluster, None
    if spec['mode'] == 'instruction_only':
        return 'instruction', None
    if registered:
        return cluster, None
    return 'instruction', f'instruction:phase_unregistered:{cluster}'
