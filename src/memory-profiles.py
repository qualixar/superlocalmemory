#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
SuperLocalMemory V2 - Profile Management System (Column-Based)

v2.4.0: Rewritten to use column-based profiles in a SINGLE database.
All memories live in one memory.db with a 'profile' column.
Switching profiles = updating config. No file copying. No data loss risk.

Previous versions used separate database files per profile, which caused
data loss when switching. This version is backward compatible and will
auto-migrate old profile directories on first run.

Allows users to maintain separate memory contexts:
- Work profile: Professional coding memories
- Personal profile: Personal projects and learning
- Client-specific profiles: Different clients get isolated memories
- Experimentation profile: Testing and experiments
"""

import os
import sys
import json
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
import argparse
import re

MEMORY_DIR = Path.home() / ".claude-memory"
DB_PATH = MEMORY_DIR / "memory.db"
PROFILES_DIR = MEMORY_DIR / "profiles"
CONFIG_FILE = MEMORY_DIR / "profiles.json"


class ProfileManager:
    """
    Column-based profile manager. All memories in ONE database.
    Profile = a value in the 'profile' column of the memories table.
    Switching = updating which profile name is active in config.
    """

    def __init__(self):
        self.memory_dir = MEMORY_DIR
        self.db_path = DB_PATH
        self.config_file = CONFIG_FILE

        # Ensure memory directory exists
        self.memory_dir.mkdir(exist_ok=True)

        # Ensure profile column exists in DB
        self._ensure_profile_column()

        # Load or create config
        self.config = self._load_config()

        # Auto-migrate old profile directories if they exist
        self._migrate_old_profiles()

    def _ensure_profile_column(self):
        """Add 'profile' column to memories table if it doesn't exist."""
        if not self.db_path.exists():
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cursor.fetchall()}

        if 'profile' not in columns:
            cursor.execute("ALTER TABLE memories ADD COLUMN profile TEXT DEFAULT 'default'")
            cursor.execute("UPDATE memories SET profile = 'default' WHERE profile IS NULL")
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_profile ON memories(profile)")
            except sqlite3.OperationalError:
                pass
            conn.commit()

        conn.close()

    def _migrate_old_profiles(self):
        """
        Backward compatibility: migrate old separate-DB profiles into the main DB.
        Old profiles stored in ~/.claude-memory/profiles/<name>/memory.db
        New profiles use a 'profile' column in the main memory.db
        """
        profiles_dir = MEMORY_DIR / "profiles"
        if not profiles_dir.exists():
            return

        migrated_any = False
        for profile_dir in profiles_dir.iterdir():
            if not profile_dir.is_dir():
                continue

            profile_db = profile_dir / "memory.db"
            if not profile_db.exists():
                continue

            profile_name = profile_dir.name
            marker_file = profile_dir / ".migrated_to_column"

            # Skip if already migrated
            if marker_file.exists():
                continue

            # Import memories from old profile DB
            try:
                self._import_from_old_profile(profile_name, profile_db)
                # Mark as migrated
                marker_file.write_text(datetime.now().isoformat())
                migrated_any = True
            except Exception as e:
                print(f"  Warning: Could not migrate profile '{profile_name}': {e}", file=sys.stderr)

        if migrated_any:
            print("  Profile migration complete (old separate DBs -> column-based)", file=sys.stderr)

    def _import_from_old_profile(self, profile_name, old_db_path):
        """Import memories from an old separate-DB profile into main DB."""
        if not self.db_path.exists():
            return

        main_conn = sqlite3.connect(self.db_path)
        main_cursor = main_conn.cursor()

        old_conn = sqlite3.connect(old_db_path)
        try:
            old_cursor = old_conn.cursor()

            # Get existing hashes
            main_cursor.execute("SELECT content_hash FROM memories WHERE content_hash IS NOT NULL")
            existing_hashes = {row[0] for row in main_cursor.fetchall()}

            # Get columns from old DB
            old_cursor.execute("PRAGMA table_info(memories)")
            old_columns = {row[1] for row in old_cursor.fetchall()}

            # Build SELECT based on available columns
            select_cols = ['content', 'summary', 'project_path', 'project_name', 'tags',
                           'category', 'memory_type', 'importance', 'created_at', 'updated_at',
                           'content_hash']
            available_cols = [c for c in select_cols if c in old_columns]

            if 'content' not in available_cols:
                return

            old_cursor.execute(f"SELECT {', '.join(available_cols)} FROM memories")
            rows = old_cursor.fetchall()

            imported = 0
            for row in rows:
                row_dict = dict(zip(available_cols, row))
                content = row_dict.get('content', '')
                content_hash = row_dict.get('content_hash')

                if not content:
                    continue

                # Generate hash if missing
                if not content_hash:
                    content_hash = hashlib.sha256(content.encode()).hexdigest()[:32]

                if content_hash in existing_hashes:
                    continue

                try:
                    main_cursor.execute('''
                        INSERT INTO memories (content, summary, project_path, project_name, tags,
                                              category, memory_type, importance, created_at, updated_at,
                                              content_hash, profile)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        content,
                        row_dict.get('summary'),
                        row_dict.get('project_path'),
                        row_dict.get('project_name'),
                        row_dict.get('tags'),
                        row_dict.get('category'),
                        row_dict.get('memory_type', 'session'),
                        row_dict.get('importance', 5),
                        row_dict.get('created_at'),
                        row_dict.get('updated_at'),
                        content_hash,
                        profile_name
                    ))
                    imported += 1
                    existing_hashes.add(content_hash)
                except sqlite3.IntegrityError:
                    pass

            main_conn.commit()
        finally:
            old_conn.close()
            main_conn.close()

        if imported > 0:
            # Add profile to config if not present
            config = self._load_config()
            if profile_name not in config.get('profiles', {}):
                config['profiles'][profile_name] = {
                    'name': profile_name,
                    'description': f'Memory profile: {profile_name} (migrated)',
                    'created_at': datetime.now().isoformat(),
                    'last_used': None
                }
                self._save_config(config)

    def _load_config(self):
        """Load profiles configuration."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            config = {
                'profiles': {
                    'default': {
                        'name': 'default',
                        'description': 'Default memory profile',
                        'created_at': datetime.now().isoformat(),
                        'last_used': datetime.now().isoformat()
                    }
                },
                'active_profile': 'default'
            }
            self._save_config(config)
            return config

    def _save_config(self, config=None):
        """Save profiles configuration."""
        if config is None:
            config = self.config

        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def _validate_profile_name(self, profile_name):
        """Validate profile name for security."""
        if not profile_name:
            raise ValueError("Profile name cannot be empty")

        if not re.match(r'^[a-zA-Z0-9_-]+$', profile_name):
            raise ValueError("Invalid profile name. Use only letters, numbers, dash, underscore.")

        if len(profile_name) > 50:
            raise ValueError("Profile name too long (max 50 characters)")

        if profile_name in ['.', '..']:
            raise ValueError(f"Reserved profile name: {profile_name}")

    def _get_memory_count(self, profile_name):
        """Get memory count for a specific profile."""
        if not self.db_path.exists():
            return 0

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories WHERE profile = ?", (profile_name,))
            count = cursor.fetchone()[0]
        finally:
            conn.close()
        return count

    def get_active_profile(self) -> str:
        """Get the currently active profile name."""
        return self.config.get('active_profile', 'default')

    def list_profiles(self) -> list:
        """List all available profiles with memory counts."""
        print("\n" + "=" * 60)
        print("AVAILABLE MEMORY PROFILES")
        print("=" * 60)

        active = self.config.get('active_profile', 'default')

        if not self.config.get('profiles'):
            print("\n  No profiles found. Create one with: create <name>")
            return

        print(f"\n{'Profile':20s} {'Description':30s} {'Memories':10s} {'Status':10s}")
        print("-" * 75)

        for name, info in self.config['profiles'].items():
            status = "ACTIVE" if name == active else ""
            desc = info.get('description', 'No description')[:30]
            marker = "-> " if name == active else "   "
            count = self._get_memory_count(name)
            print(f"{marker}{name:17s} {desc:30s} {count:<10d} {status:10s}")

        print(f"\nTotal profiles: {len(self.config['profiles'])}")
        print(f"Active profile: {active}")

    def create_profile(self, name, description=None, from_current=False):
        """
        Create a new profile.
        Column-based: just adds to config. If from_current, copies memories.
        """
        print(f"\nCreating profile: {name}")

        self._validate_profile_name(name)

        if name in self.config['profiles']:
            print(f"Error: Profile '{name}' already exists")
            return False

        if from_current:
            # Copy current profile's memories to new profile
            active = self.config.get('active_profile', 'default')
            if self.db_path.exists():
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO memories (content, summary, project_path, project_name, tags,
                                          category, parent_id, tree_path, depth, memory_type,
                                          importance, created_at, updated_at, last_accessed,
                                          access_count, content_hash, cluster_id, profile)
                    SELECT content, summary, project_path, project_name, tags,
                           category, parent_id, tree_path, depth, memory_type,
                           importance, created_at, updated_at, last_accessed,
                           access_count, NULL, cluster_id, ?
                    FROM memories WHERE profile = ?
                """, (name, active))
                copied = cursor.rowcount
                conn.commit()

                # Generate new content hashes for copied memories
                cursor.execute("SELECT id, content FROM memories WHERE profile = ? AND content_hash IS NULL", (name,))
                for row in cursor.fetchall():
                    new_hash = hashlib.sha256(row[1].encode()).hexdigest()[:32]
                    try:
                        cursor.execute("UPDATE memories SET content_hash = ? WHERE id = ?", (new_hash + f"_{name}", row[0]))
                    except sqlite3.IntegrityError:
                        pass
                conn.commit()
                conn.close()

                print(f"  Copied {copied} memories from '{active}' profile")
            else:
                print("  No database found, creating empty profile")
        else:
            print(f"  Empty profile created (memories will be saved here when active)")

        # Add to config
        self.config['profiles'][name] = {
            'name': name,
            'description': description or f'Memory profile: {name}',
            'created_at': datetime.now().isoformat(),
            'last_used': None,
            'created_from': 'current' if from_current else 'empty'
        }
        self._save_config()

        print(f"Profile '{name}' created successfully")
        return True

    def switch_profile(self, name):
        """
        Switch to a different profile.
        Column-based: just updates the active_profile in config. Instant. No data risk.
        """
        if name not in self.config['profiles']:
            print(f"Error: Profile '{name}' not found")
            print(f"   Available: {', '.join(self.config['profiles'].keys())}")
            return False

        current = self.config.get('active_profile', 'default')

        if current == name:
            print(f"Already using profile: {name}")
            return True

        # Column-based switch: just update config
        self.config['active_profile'] = name
        self.config['profiles'][name]['last_used'] = datetime.now().isoformat()
        self._save_config()

        count = self._get_memory_count(name)
        print(f"\nSwitched to profile: {name} ({count} memories)")
        print(f"Previous profile: {current}")
        return True

    def delete_profile(self, name, force=False):
        """Delete a profile. Moves memories to 'default' or deletes them."""
        if name not in self.config['profiles']:
            print(f"Error: Profile '{name}' not found")
            return False

        if name == 'default':
            print(f"Error: Cannot delete 'default' profile")
            return False

        if self.config.get('active_profile') == name:
            print(f"Error: Cannot delete active profile")
            print(f"   Switch to another profile first: slm profile switch default")
            return False

        count = self._get_memory_count(name)

        if not force:
            print(f"\nWARNING: This will delete profile '{name}' ({count} memories)")
            print(f"Memories will be moved to 'default' profile before deletion.")
            response = input(f"Type profile name '{name}' to confirm: ")

            if response != name:
                print("Cancelled.")
                return False

        # Move memories to default profile
        if self.db_path.exists() and count > 0:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("UPDATE memories SET profile = 'default' WHERE profile = ?", (name,))
                moved = cursor.rowcount
                conn.commit()
            finally:
                conn.close()
            print(f"  Moved {moved} memories to 'default' profile")

        # Remove from config
        del self.config['profiles'][name]
        self._save_config()

        print(f"Profile '{name}' deleted")
        return True

    def show_current(self):
        """Show current active profile with stats."""
        active = self.config.get('active_profile', 'default')

        if active in self.config['profiles']:
            info = self.config['profiles'][active]

            print("\n" + "=" * 60)
            print("CURRENT ACTIVE PROFILE")
            print("=" * 60)
            print(f"\nProfile: {active}")
            print(f"Description: {info.get('description', 'N/A')}")
            print(f"Created: {info.get('created_at', 'N/A')}")
            print(f"Last used: {info.get('last_used', 'N/A')}")

            count = self._get_memory_count(active)
            print(f"\nMemories in this profile: {count}")

            # Show total memories across all profiles
            if self.db_path.exists():
                conn = sqlite3.connect(self.db_path)
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM memories")
                    total = cursor.fetchone()[0]
                finally:
                    conn.close()
                print(f"Total memories (all profiles): {total}")
        else:
            print(f"Warning: Current profile '{active}' not found in config")
            print("Resetting to 'default' profile...")
            self.config['active_profile'] = 'default'
            self._save_config()

    def rename_profile(self, old_name, new_name):
        """Rename a profile. Updates the column value in all memories."""
        if old_name not in self.config['profiles']:
            print(f"Error: Profile '{old_name}' not found")
            return False

        if new_name in self.config['profiles']:
            print(f"Error: Profile '{new_name}' already exists")
            return False

        if old_name == 'default':
            print(f"Error: Cannot rename 'default' profile")
            return False

        self._validate_profile_name(new_name)

        # Update profile column in all memories
        if self.db_path.exists():
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("UPDATE memories SET profile = ? WHERE profile = ?", (new_name, old_name))
                updated = cursor.rowcount
                conn.commit()
            finally:
                conn.close()
            print(f"  Updated {updated} memories")

        # Update config
        self.config['profiles'][new_name] = self.config['profiles'][old_name]
        self.config['profiles'][new_name]['name'] = new_name
        del self.config['profiles'][old_name]

        if self.config.get('active_profile') == old_name:
            self.config['active_profile'] = new_name

        self._save_config()

        print(f"Profile renamed: '{old_name}' -> '{new_name}'")
        return True



def main():
    parser = argparse.ArgumentParser(
        description='SuperLocalMemory V2 - Profile Management (Column-Based)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # List all profiles
  python memory-profiles.py list

  # Show current profile
  python memory-profiles.py current

  # Create new empty profile
  python memory-profiles.py create work --description "Work projects"

  # Create profile from current memories
  python memory-profiles.py create personal --from-current

  # Switch to different profile (instant, no restart needed)
  python memory-profiles.py switch work

  # Delete a profile (memories moved to default)
  python memory-profiles.py delete old-profile

  # Rename profile
  python memory-profiles.py rename old-name new-name

Architecture (v2.4.0):
  All profiles share ONE database (memory.db).
  Each memory has a 'profile' column.
  Switching profiles = changing config. Instant. Safe.
        '''
    )

    parser.add_argument('command',
                       choices=['list', 'current', 'create', 'switch', 'delete', 'rename'],
                       help='Profile command')
    parser.add_argument('name', nargs='?', help='Profile name')
    parser.add_argument('name2', nargs='?', help='Second name (for rename)')
    parser.add_argument('--description', help='Profile description')
    parser.add_argument('--from-current', action='store_true',
                       help='Create from current profile memories')
    parser.add_argument('--force', action='store_true',
                       help='Force operation without confirmation')

    args = parser.parse_args()

    manager = ProfileManager()

    if args.command == 'list':
        manager.list_profiles()

    elif args.command == 'current':
        manager.show_current()

    elif args.command == 'create':
        if not args.name:
            print("Error: Profile name required")
            print("   Usage: python memory-profiles.py create <name>")
            sys.exit(1)

        manager.create_profile(args.name, args.description, args.from_current)

    elif args.command == 'switch':
        if not args.name:
            print("Error: Profile name required")
            sys.exit(1)

        manager.switch_profile(args.name)

    elif args.command == 'delete':
        if not args.name:
            print("Error: Profile name required")
            sys.exit(1)

        manager.delete_profile(args.name, args.force)

    elif args.command == 'rename':
        if not args.name or not args.name2:
            print("Error: Both old and new names required")
            print("   Usage: python memory-profiles.py rename <old> <new>")
            sys.exit(1)

        manager.rename_profile(args.name, args.name2)


if __name__ == '__main__':
    main()
