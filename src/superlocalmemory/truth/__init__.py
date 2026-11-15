# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The truth layer (4.2.30 WS-1): slots, revisions, disputes, validity and leases.

It decides which memory is the current value of something, keeps every older
value as history, shows disagreements as disputes instead of silently picking
one, and notices when the source a memory came from has changed. Every effect
on a fact's validity goes through the correction ledger, so it can be rolled
back exactly.
"""
