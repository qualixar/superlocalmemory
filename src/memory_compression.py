#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Progressive Summarization Compression for SuperLocalMemory
Tier-based compression system to maintain 100+ memories efficiently.

Tier Strategy:
- Tier 1 (0-30 days): Full content (no compression)
- Tier 2 (30-90 days): Summary + key excerpts (~80% reduction)
- Tier 3 (90+ days): Bullet points only (~96% reduction)
- Cold Storage (1+ year): Gzipped JSON archives (~98% reduction)

No external LLM calls - all compression is extractive using local algorithms.
"""

import sqlite3
import json
import gzip
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import hashlib


MEMORY_DIR = Path.home() / ".claude-memory"
DB_PATH = MEMORY_DIR / "memory.db"
CONFIG_PATH = MEMORY_DIR / "config.json"
COLD_STORAGE_PATH = MEMORY_DIR / "cold-storage"
LOGS_PATH = MEMORY_DIR / "logs"


class CompressionConfig:
    """Configuration for compression behavior."""

    def __init__(self):
        self.config = self._load_config()
        self.compression_settings = self.config.get('compression', {})

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json."""
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        return {}

    def save(self) -> None:
        """Save configuration back to config.json."""
        with open(CONFIG_PATH, 'w') as f:
            json.dump(self.config, f, indent=2)

    @property
    def enabled(self) -> bool:
        return self.compression_settings.get('enabled', True)

    @property
    def tier2_threshold_days(self) -> int:
        return self.compression_settings.get('tier2_threshold_days', 30)

    @property
    def tier3_threshold_days(self) -> int:
        return self.compression_settings.get('tier3_threshold_days', 90)

    @property
    def cold_storage_threshold_days(self) -> int:
        return self.compression_settings.get('cold_storage_threshold_days', 365)

    @property
    def preserve_high_importance(self) -> bool:
        return self.compression_settings.get('preserve_high_importance', True)

    @property
    def preserve_recently_accessed(self) -> bool:
        return self.compression_settings.get('preserve_recently_accessed', True)

    def initialize_defaults(self) -> None:
        """Initialize compression settings in config if not present."""
        if 'compression' not in self.config:
            self.config['compression'] = {
                'enabled': True,
                'tier2_threshold_days': 30,
                'tier3_threshold_days': 90,
                'cold_storage_threshold_days': 365,
                'preserve_high_importance': True,
                'preserve_recently_accessed': True
            }
            self.save()


class TierClassifier:
    """Classify memories into compression tiers based on age and access patterns."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.config = CompressionConfig()
        self._ensure_schema()

    def _ensure_schema(self):
        """Add tier and access tracking columns if not present."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if tier column exists
        cursor.execute("PRAGMA table_info(memories)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'tier' not in columns:
            cursor.execute('ALTER TABLE memories ADD COLUMN tier INTEGER DEFAULT 1')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tier ON memories(tier)')

        if 'last_accessed' not in columns:
            cursor.execute('ALTER TABLE memories ADD COLUMN last_accessed TIMESTAMP')

        if 'access_count' not in columns:
            cursor.execute('ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0')

        # Create memory_archive table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_archive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER UNIQUE NOT NULL,
                full_content TEXT NOT NULL,
                archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_archive_memory ON memory_archive(memory_id)')

        conn.commit()
        conn.close()

    def classify_memories(self) -> List[Tuple[int, int]]:
        """
        Classify all memories into tiers based on age and access.

        Returns:
            List of (tier, memory_id) tuples
        """
        if not self.config.enabled:
            return []

        now = datetime.now()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all memories with access tracking
        cursor.execute('''
            SELECT id, created_at, last_accessed, access_count, importance, tier
            FROM memories
        ''')
        memories = cursor.fetchall()

        tier_updates = []

        for memory_id, created_at, last_accessed, access_count, importance, current_tier in memories:
            created = datetime.fromisoformat(created_at)
            age_days = (now - created).days

            # Override: High-importance memories stay in Tier 1
            if self.config.preserve_high_importance and importance and importance >= 8:
                tier = 1
            # Recently accessed stays in Tier 1
            elif self.config.preserve_recently_accessed and last_accessed:
                last_access = datetime.fromisoformat(last_accessed)
                if (now - last_access).days < 7:
                    tier = 1
                else:
                    tier = self._classify_by_age(age_days)
            # Age-based classification
            else:
                tier = self._classify_by_age(age_days)

            # Only update if tier changed
            if tier != current_tier:
                tier_updates.append((tier, memory_id))

        # Update tier field
        if tier_updates:
            cursor.executemany('''
                UPDATE memories SET tier = ? WHERE id = ?
            ''', tier_updates)
            conn.commit()

        conn.close()
        return tier_updates

    def _classify_by_age(self, age_days: int) -> int:
        """Classify memory tier based on age."""
        if age_days < self.config.tier2_threshold_days:
            return 1  # Recent
        elif age_days < self.config.tier3_threshold_days:
            return 2  # Active
        else:
            return 3  # Archived

    def get_tier_stats(self) -> Dict[str, int]:
        """Get count of memories in each tier."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT tier, COUNT(*) FROM memories GROUP BY tier
        ''')
        stats = dict(cursor.fetchall())
        conn.close()

        return {
            'tier1': stats.get(1, 0),
            'tier2': stats.get(2, 0),
            'tier3': stats.get(3, 0)
        }


class Tier2Compressor:
    """Compress memories to summary + key excerpts (Tier 2)."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def compress_to_tier2(self, memory_id: int) -> bool:
        """
        Compress memory to summary + excerpts.

        Args:
            memory_id: ID of memory to compress

        Returns:
            True if compression succeeded, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get full content
        cursor.execute('''
            SELECT content, summary, tier FROM memories WHERE id = ?
        ''', (memory_id,))
        result = cursor.fetchone()

        if not result:
            conn.close()
            return False

        content, existing_summary, current_tier = result

        # Skip if already compressed or in wrong tier
        if current_tier != 2:
            conn.close()
            return False

        # Check if already archived (don't re-compress)
        cursor.execute('''
            SELECT full_content FROM memory_archive WHERE memory_id = ?
        ''', (memory_id,))
        if cursor.fetchone():
            conn.close()
            return True  # Already compressed

        # Try to parse as JSON (might already be compressed)
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and 'summary' in parsed:
                conn.close()
                return True  # Already compressed
        except (json.JSONDecodeError, TypeError):
            pass  # Not compressed yet

        # Generate/enhance summary if needed
        if not existing_summary or len(existing_summary) < 100:
            summary = self._generate_summary(content)
        else:
            summary = existing_summary

        # Extract key excerpts (important sentences, code blocks, lists)
        excerpts = self._extract_key_excerpts(content)

        # Store compressed version
        compressed_content = {
            'summary': summary,
            'excerpts': excerpts,
            'original_length': len(content),
            'compressed_at': datetime.now().isoformat()
        }

        # Move full content to archive table
        cursor.execute('''
            INSERT INTO memory_archive (memory_id, full_content, archived_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (memory_id, content))

        # Update memory with compressed version
        cursor.execute('''
            UPDATE memories
            SET content = ?, tier = 2, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (json.dumps(compressed_content), memory_id))

        conn.commit()
        conn.close()
        return True

    def _generate_summary(self, content: str, max_length: int = 300) -> str:
        """
        Generate extractive summary from content.
        Uses sentence scoring based on heuristics (no external LLM).

        Args:
            content: Full content text
            max_length: Maximum summary length in characters

        Returns:
            Extracted summary
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)

        # Score sentences by importance (simple heuristic)
        scored_sentences = []

        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if len(sent) < 10:
                continue

            score = 0

            # Boost if contains tech terms
            tech_terms = ['api', 'database', 'auth', 'component', 'function',
                         'class', 'method', 'variable', 'error', 'bug', 'fix',
                         'implement', 'refactor', 'test', 'deploy']
            score += sum(1 for term in tech_terms if term in sent.lower())

            # Boost if at start or end (thesis/conclusion)
            if i == 0 or i == len(sentences) - 1:
                score += 2

            # Boost if contains numbers/specifics
            if re.search(r'\d+', sent):
                score += 1

            # Boost if contains important keywords
            important_keywords = ['important', 'critical', 'note', 'remember',
                                 'key', 'main', 'primary', 'must', 'should']
            score += sum(2 for kw in important_keywords if kw in sent.lower())

            scored_sentences.append((score, sent))

        # Take top sentences up to max_length
        scored_sentences.sort(reverse=True, key=lambda x: x[0])

        summary_parts = []
        current_length = 0

        for score, sent in scored_sentences:
            if current_length + len(sent) > max_length:
                break

            summary_parts.append(sent)
            current_length += len(sent)

        if not summary_parts:
            # Fallback: take first sentence
            return sentences[0][:max_length] if sentences else content[:max_length]

        return '. '.join(summary_parts) + '.'

    def _extract_key_excerpts(self, content: str, max_excerpts: int = 3) -> List[str]:
        """
        Extract key excerpts (code blocks, lists, important paragraphs).

        Args:
            content: Full content text
            max_excerpts: Maximum number of excerpts to extract

        Returns:
            List of excerpt strings
        """
        excerpts = []

        # Extract code blocks (markdown or indented)
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        excerpts.extend(code_blocks[:2])  # Max 2 code blocks

        # Extract bullet lists
        list_pattern = r'(?:^|\n)(?:[-*•]|\d+\.)\s+.+(?:\n(?:[-*•]|\d+\.)\s+.+)*'
        lists = re.findall(list_pattern, content, re.MULTILINE)
        if lists and len(excerpts) < max_excerpts:
            excerpts.extend(lists[:1])  # Max 1 list

        # Extract paragraphs with important keywords if we need more
        if len(excerpts) < max_excerpts:
            paragraphs = content.split('\n\n')
            important_keywords = ['important', 'critical', 'note', 'remember', 'key']

            for para in paragraphs:
                if len(excerpts) >= max_excerpts:
                    break

                if any(kw in para.lower() for kw in important_keywords):
                    # Truncate long paragraphs
                    if len(para) > 200:
                        para = para[:197] + '...'
                    excerpts.append(para)

        # Truncate if too many
        return excerpts[:max_excerpts]

    def compress_all_tier2(self) -> int:
        """Compress all memories that are in Tier 2."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT id FROM memories WHERE tier = 2')
        memory_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        compressed_count = 0
        for memory_id in memory_ids:
            if self.compress_to_tier2(memory_id):
                compressed_count += 1

        return compressed_count


class Tier3Compressor:
    """Compress memories to bullet points only (Tier 3)."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def compress_to_tier3(self, memory_id: int) -> bool:
        """
        Compress memory to bullet points only.

        Args:
            memory_id: ID of memory to compress

        Returns:
            True if compression succeeded, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get Tier 2 compressed content
        cursor.execute('''
            SELECT content, tier FROM memories WHERE id = ?
        ''', (memory_id,))
        result = cursor.fetchone()

        if not result:
            conn.close()
            return False

        content, current_tier = result

        # Skip if in wrong tier
        if current_tier != 3:
            conn.close()
            return False

        # Try to parse as Tier 2 compressed content
        try:
            compressed_content = json.loads(content)

            # Check if already Tier 3
            if isinstance(compressed_content, dict) and 'bullets' in compressed_content:
                conn.close()
                return True  # Already Tier 3

            # Get summary from Tier 2
            if isinstance(compressed_content, dict) and 'summary' in compressed_content:
                summary = compressed_content.get('summary', '')
                tier2_archived_at = compressed_content.get('compressed_at')
                original_length = compressed_content.get('original_length', 0)
            else:
                # Not Tier 2 format, treat as plain text
                summary = content
                tier2_archived_at = None
                original_length = len(content)

        except (json.JSONDecodeError, TypeError):
            # Not JSON, treat as plain text
            summary = content
            tier2_archived_at = None
            original_length = len(content)

        # Convert summary to bullet points (max 5)
        bullet_points = self._summarize_to_bullets(summary)

        # Ultra-compressed version
        ultra_compressed = {
            'bullets': bullet_points,
            'tier2_archived_at': tier2_archived_at,
            'original_length': original_length,
            'compressed_to_tier3_at': datetime.now().isoformat()
        }

        # Update memory
        cursor.execute('''
            UPDATE memories
            SET content = ?, tier = 3, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (json.dumps(ultra_compressed), memory_id))

        conn.commit()
        conn.close()
        return True

    def _summarize_to_bullets(self, summary: str, max_bullets: int = 5) -> List[str]:
        """
        Convert summary to bullet points.

        Args:
            summary: Summary text
            max_bullets: Maximum number of bullets

        Returns:
            List of bullet point strings
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', summary)

        bullets = []

        for sent in sentences:
            sent = sent.strip()

            if len(sent) < 10:
                continue

            # Truncate long sentences
            if len(sent) > 80:
                sent = sent[:77] + '...'

            bullets.append(sent)

            if len(bullets) >= max_bullets:
                break

        return bullets if bullets else ['[No summary available]']

    def compress_all_tier3(self) -> int:
        """Compress all memories that are in Tier 3."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT id FROM memories WHERE tier = 3')
        memory_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        compressed_count = 0
        for memory_id in memory_ids:
            if self.compress_to_tier3(memory_id):
                compressed_count += 1

        return compressed_count


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
            conn.close()
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
                                cursor = conn.cursor()

                                cursor.execute('''
                                    INSERT OR REPLACE INTO memory_archive
                                    (memory_id, full_content, archived_at)
                                    VALUES (?, ?, CURRENT_TIMESTAMP)
                                ''', (memory_id, full_content))

                                conn.commit()
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
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id FROM memories
            WHERE tier = 3
            AND created_at < ?
            AND importance < 8
        ''', (threshold_date.isoformat(),))

        memory_ids = [row[0] for row in cursor.fetchall()]
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


class CompressionOrchestrator:
    """Main orchestrator for compression operations."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.config = CompressionConfig()
        self.classifier = TierClassifier(db_path)
        self.tier2_compressor = Tier2Compressor(db_path)
        self.tier3_compressor = Tier3Compressor(db_path)
        self.cold_storage = ColdStorageManager(db_path)

    def run_full_compression(self) -> Dict[str, Any]:
        """
        Run full compression cycle: classify, compress, and archive.

        Returns:
            Statistics about compression operation
        """
        if not self.config.enabled:
            return {'status': 'disabled', 'message': 'Compression is disabled in config'}

        stats = {
            'started_at': datetime.now().isoformat(),
            'tier_updates': 0,
            'tier2_compressed': 0,
            'tier3_compressed': 0,
            'cold_stored': 0,
            'errors': []
        }

        try:
            # Step 1: Classify memories into tiers
            tier_updates = self.classifier.classify_memories()
            stats['tier_updates'] = len(tier_updates)

            # Step 2: Compress Tier 2 memories
            stats['tier2_compressed'] = self.tier2_compressor.compress_all_tier2()

            # Step 3: Compress Tier 3 memories
            stats['tier3_compressed'] = self.tier3_compressor.compress_all_tier3()

            # Step 4: Move old memories to cold storage
            candidates = self.cold_storage.get_cold_storage_candidates()
            if candidates:
                stats['cold_stored'] = self.cold_storage.move_to_cold_storage(candidates)

            # Get final tier stats
            stats['tier_stats'] = self.classifier.get_tier_stats()

            # Calculate space savings
            stats['space_savings'] = self._calculate_space_savings()

        except Exception as e:
            stats['errors'].append(str(e))

        stats['completed_at'] = datetime.now().isoformat()
        return stats

    def _calculate_space_savings(self) -> Dict[str, Any]:
        """Calculate estimated space savings from compression."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get size of compressed content
        cursor.execute('''
            SELECT
                tier,
                COUNT(*) as count,
                SUM(LENGTH(content)) as total_size
            FROM memories
            GROUP BY tier
        ''')

        tier_sizes = {}
        for tier, count, total_size in cursor.fetchall():
            tier_sizes[tier] = {
                'count': count,
                'size_bytes': total_size or 0
            }

        # Get size of archived content
        cursor.execute('''
            SELECT
                COUNT(*) as count,
                SUM(LENGTH(full_content)) as total_size
            FROM memory_archive
        ''')
        archive_count, archive_size = cursor.fetchone()

        conn.close()

        # Estimate original size if all were Tier 1
        tier1_avg = tier_sizes.get(1, {}).get('size_bytes', 50000) / max(tier_sizes.get(1, {}).get('count', 1), 1)
        total_memories = sum(t.get('count', 0) for t in tier_sizes.values())
        estimated_original = int(tier1_avg * total_memories)

        current_size = sum(t.get('size_bytes', 0) for t in tier_sizes.values())

        return {
            'estimated_original_bytes': estimated_original,
            'current_size_bytes': current_size,
            'savings_bytes': estimated_original - current_size,
            'savings_percent': round((1 - current_size / max(estimated_original, 1)) * 100, 1),
            'tier_breakdown': tier_sizes,
            'archive_count': archive_count or 0,
            'archive_size_bytes': archive_size or 0
        }


# CLI Interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Progressive Summarization Compression for SuperLocalMemory\n")
        print("Usage:")
        print("  python compression.py classify         # Classify memories into tiers")
        print("  python compression.py compress         # Run full compression cycle")
        print("  python compression.py stats            # Show compression statistics")
        print("  python compression.py tier2 <id>       # Compress specific memory to Tier 2")
        print("  python compression.py tier3 <id>       # Compress specific memory to Tier 3")
        print("  python compression.py cold-storage     # Move old memories to cold storage")
        print("  python compression.py restore <id>     # Restore memory from cold storage")
        print("  python compression.py init-config      # Initialize compression config")
        sys.exit(0)

    command = sys.argv[1]
    orchestrator = CompressionOrchestrator()

    if command == "classify":
        classifier = TierClassifier()
        updates = classifier.classify_memories()
        print(f"Classified {len(updates)} memories")

        stats = classifier.get_tier_stats()
        print(f"\nTier breakdown:")
        print(f"  Tier 1 (Full content):    {stats['tier1']} memories")
        print(f"  Tier 2 (Summary+excerpts): {stats['tier2']} memories")
        print(f"  Tier 3 (Bullets only):     {stats['tier3']} memories")

    elif command == "compress":
        print("Running full compression cycle...")
        stats = orchestrator.run_full_compression()

        print(f"\nCompression Results:")
        print(f"  Tier updates:          {stats['tier_updates']}")
        print(f"  Tier 2 compressed:     {stats['tier2_compressed']}")
        print(f"  Tier 3 compressed:     {stats['tier3_compressed']}")
        print(f"  Moved to cold storage: {stats['cold_stored']}")

        if 'space_savings' in stats:
            savings = stats['space_savings']
            print(f"\nSpace Savings:")
            print(f"  Original size:  {savings['estimated_original_bytes']:,} bytes")
            print(f"  Current size:   {savings['current_size_bytes']:,} bytes")
            print(f"  Savings:        {savings['savings_bytes']:,} bytes ({savings['savings_percent']}%)")

        if stats.get('errors'):
            print(f"\nErrors: {stats['errors']}")

    elif command == "stats":
        classifier = TierClassifier()
        tier_stats = classifier.get_tier_stats()

        cold_storage = ColdStorageManager()
        cold_stats = cold_storage.get_cold_storage_stats()

        savings = orchestrator._calculate_space_savings()

        print("Compression Statistics\n")
        print("Tier Breakdown:")
        print(f"  Tier 1 (Full content):     {tier_stats['tier1']} memories")
        print(f"  Tier 2 (Summary+excerpts): {tier_stats['tier2']} memories")
        print(f"  Tier 3 (Bullets only):     {tier_stats['tier3']} memories")

        print(f"\nCold Storage:")
        print(f"  Archive files: {cold_stats['archive_count']}")
        print(f"  Total memories: {cold_stats['total_memories']}")
        print(f"  Total size: {cold_stats['total_size_bytes']:,} bytes")

        print(f"\nSpace Savings:")
        print(f"  Estimated original: {savings['estimated_original_bytes']:,} bytes")
        print(f"  Current size:       {savings['current_size_bytes']:,} bytes")
        print(f"  Savings:            {savings['savings_bytes']:,} bytes ({savings['savings_percent']}%)")

    elif command == "tier2" and len(sys.argv) >= 3:
        try:
            memory_id = int(sys.argv[2])
            compressor = Tier2Compressor()
            if compressor.compress_to_tier2(memory_id):
                print(f"Memory #{memory_id} compressed to Tier 2")
            else:
                print(f"Failed to compress memory #{memory_id}")
        except ValueError:
            print("Error: Memory ID must be a number")

    elif command == "tier3" and len(sys.argv) >= 3:
        try:
            memory_id = int(sys.argv[2])
            compressor = Tier3Compressor()
            if compressor.compress_to_tier3(memory_id):
                print(f"Memory #{memory_id} compressed to Tier 3")
            else:
                print(f"Failed to compress memory #{memory_id}")
        except ValueError:
            print("Error: Memory ID must be a number")

    elif command == "cold-storage":
        cold_storage = ColdStorageManager()
        candidates = cold_storage.get_cold_storage_candidates()

        if not candidates:
            print("No memories ready for cold storage")
        else:
            print(f"Moving {len(candidates)} memories to cold storage...")
            count = cold_storage.move_to_cold_storage(candidates)
            print(f"Archived {count} memories")

    elif command == "restore" and len(sys.argv) >= 3:
        try:
            memory_id = int(sys.argv[2])
            cold_storage = ColdStorageManager()
            content = cold_storage.restore_from_cold_storage(memory_id)

            if content:
                print(f"Memory #{memory_id} restored from cold storage")
            else:
                print(f"Memory #{memory_id} not found in cold storage")
        except ValueError:
            print("Error: Memory ID must be a number")

    elif command == "init-config":
        config = CompressionConfig()
        config.initialize_defaults()
        print("Compression configuration initialized")
        print(json.dumps(config.compression_settings, indent=2))

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
