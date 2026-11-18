# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The memory compiler (4.2.30, WS-6): versions of derived artifacts and ProposalSets.

``versions`` keeps the previous row of every derived artifact before it is
overwritten. ``proposals`` is the unit of proposed change with its basis and its
exact rollback. Nothing in this package changes a memory without a basis (I-10).
"""
