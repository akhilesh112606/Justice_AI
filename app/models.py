"""Lightweight SQLite helpers for the criminals image store."""

import sqlite3
from pathlib import Path


def get_db_path() -> Path:
	"""Return the path to the SQLite database file."""
	root = Path(__file__).resolve().parent.parent
	db_dir = root / "database"
	db_dir.mkdir(parents=True, exist_ok=True)
	return db_dir / "data.db"


def init_db():
	"""Create the criminals table if it does not already exist."""
	db_path = get_db_path()
	with sqlite3.connect(db_path) as conn:
		cur = conn.cursor()
		cur.execute(
			"""
			CREATE TABLE IF NOT EXISTS criminals (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				name TEXT NOT NULL,
				alias TEXT,
				image BLOB NOT NULL,
				mime_type TEXT DEFAULT 'image/jpeg',
				notes TEXT,
				created_at TEXT DEFAULT (datetime('now'))
			);
			"""
		)
		conn.commit()
	return db_path


# Ensure the database exists when this module loads.
DB_PATH = init_db()
