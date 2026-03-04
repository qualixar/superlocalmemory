#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
CLI interface for compression operations.
"""

import sys
import json

from compression.config import CompressionConfig
from compression.tier_classifier import TierClassifier
from compression.tier2_compressor import Tier2Compressor
from compression.tier3_compressor import Tier3Compressor
from compression.cold_storage import ColdStorageManager
from compression.orchestrator import CompressionOrchestrator


def main():
    """Main CLI entry point."""
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


if __name__ == "__main__":
    main()
