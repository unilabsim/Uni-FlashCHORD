"""RL training execution runtime: distributed launch, IPC, off-policy runners.

This layer may depend on ``chord_unilab.rl.contracts`` and ``chord_unilab.rl.logging``;
it must not depend on concrete algorithms, ``chord_unilab.training``,
``unilab.base``, or env construction.
"""
