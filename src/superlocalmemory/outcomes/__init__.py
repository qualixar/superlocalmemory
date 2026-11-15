# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""4.2.30 WS-3: the outcome ledger (injection receipts, randomised withholding).

``randomiser`` decides, per delivered candidate, whether it is withheld and
with what logged probability; ``receipts`` plans a delivery, applies the
decision and records it. Both are local and ids-only.
"""
