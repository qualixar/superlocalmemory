# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Saving PDFs page by page: submit, the job service, status and removal."""

from superlocalmemory.documents.runner import (
    DocumentJobService,
    start_document_jobs,
    stop_document_jobs,
)
from superlocalmemory.documents.status import job_status, remove_document
from superlocalmemory.documents.submit import DocumentReceipt, submit_document

__all__ = [
    "DocumentJobService", "DocumentReceipt", "job_status", "remove_document",
    "start_document_jobs", "stop_document_jobs", "submit_document",
]
