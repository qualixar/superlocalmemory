#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Cold storage management for archived memories.
Handles compression and archival to gzipped JSON files.
"""

import sqlite3
import json
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any

from compression.config import CompressionConfig


MEMORY_DIR = Path.home() / ".claude-memory"
DB_PATH = MEMORY_DIR / "memory.db"
COLD_STORAGE_PATH = MEMORY_DIR / "cold-storage"


class ColdStorageManager:
    """Manage cold storage archives for very old memories."""

    def __init__(self, db_path: Path = DB_PATH, storage_path: Path = COLD_STORAGE_PATH):
        self.db_path = db_path
        self.storage_path = storage_path
        self.storage_path.mkdir(exist_ok=True)
        self.config = CompressionConfig()

    def move_to_cold_storage(self, memory_ids: List[int]) -> int:
        """
        Move archived memories to gzipped JSON file.

        Args:
            memory_ids: List of memory IDs to archive

        Returns:
            Number of memories archived
        """
        if not memory_ids:
            return 0

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Build placeholders for SQL query
            placeholders = ','.join('?' * len(memory_ids))

            # Get memories from archive table
            cursor.execute(f'''
                SELECT m.id, m.content, m.summary, m.tags, m.project_name,
                       m.created_at, a.full_content
                FROM memories m
                LEFT JOIN memory_archive a ON m.id = a.memory_id
                WHERE m.id IN ({placeholders})
            ''', memory_ids)

            memories = cursor.fetchall()

            if not memories:
                return 0

            # Build JSON export
            export_data = []

            for memory in memories:
                mem_id, content, summary, tags, project_name, created_at, full_content = memory

                export_data.append({
                    'id': mem_id,
                    'tier3_content': self._safe_json_load(content),
                    'summary': summary,
                    'tags': self._safe_json_load(tags) if tags else [],
                    'project': project_name,
                    'created_at': created_at,
                    'full_content': full_content  # May be None if not archived
                })

            # Write to gzipped file
            filename = f"archive-{datetime.now().strftime('%Y-%m')}.json.gz"
            filepath = self.storage_path / filename

            # If file exists, append to it
            existing_data = []
            if filepath.exists():
                try:
                    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except Exception:
                    pass  # File might be corrupted, start fresh

            # Merge with existing data (avoid duplicates)
            existing_ids = {item['id'] for item in existing_data}
            for item in export_data:
                if item['id'] not in existing_ids:
                    existing_data.append(item)

            # Write combined data
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2)

            # Delete from archive table (keep Tier 3 version in main table)
            cursor.executemany('DELETE FROM memory_archive WHERE memory_id = ?',
                              [(mid,) for mid in memory_ids])

            conn.commit()
        finally:
            conn.close()

        return len(export_data)

    def _safe_json_load(self, data: str) -> Any:
        """Safely load JSON data."""
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data

    def restore_from_cold_storage(self, memory_id: int) -> Optional[str]:
        """
        Restore full content from cold storage archive.

        Args:
            memory_id: ID of memory to restore

        Returns:
            Full content if found, None otherwise
        """
        # Search all archive files
        for archive_file in self.storage_path.glob('archive-*.json.gz'):
            try:
                with gzip.open(archive_file, 'rt', encoding='utf-8') as f:
                    data = json.load(f)

                    for memory in data:
                        if memory['id'] == memory_id:
                            full_content = memory.get('full_content')

                            if full_content:
                                # Restore to archive table
                                conn = sqlite3.connect(self.db_path)
                                try:
                                    cursor = conn.cursor()

                                    cursor.execute('''
                                        INSERT OR REPLACE INTO memory_archive
                                        (memory_id, full_content, archived_at)
                                        VALUES (?, ?, CURRENT_TIMESTAMP)
                                    ''', (memory_id, full_content))

                                    conn.commit()
                                finally:
                                    conn.close()

                                return full_content
            except Exception as e:
                print(f"Error reading archive {archive_file}: {e}")
                continue

        return None

    def get_cold_storage_candidates(self) -> List[int]:
        """Get memory IDs that are candidates for cold storage."""
        threshold_date = datetime.now() - timedelta(days=self.config.cold_storage_threshold_days)

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id FROM memories
                WHERE tier = 3
                AND created_at < ?
                AND importance < 8
            ''', (threshold_date.isoformat(),))

            memory_ids = [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

        return memory_ids

    def get_cold_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about cold storage."""
        stats = {
            'archive_count': 0,
            'total_memories': 0,
            'total_size_bytes': 0,
            'archives': []
        }

        for archive_file in self.storage_path.glob('archive-*.json.gz'):
            try:
                size = archive_file.stat().st_size

                with gzip.open(archive_file, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                    memory_count = len(data)

                stats['archive_count'] += 1
                stats['total_memories'] += memory_count
                stats['total_size_bytes'] += size

                stats['archives'].append({
                    'filename': archive_file.name,
                    'memory_count': memory_count,
                    'size_bytes': size,
                    'size_mb': round(size / 1024 / 1024, 2)
                })
            except Exception:
                continue

        return stats
