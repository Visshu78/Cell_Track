"""
database.py - Persistent SQLite Database Engine for Cell_Track / BioTrack-X

Maintains a relational database (cell_tracking.db) storing tracking experiments,
cell event logs, morphometric measurements, kinematic trajectories, and LLM biological reports.
Includes robust field normalization and automated maintenance routines.
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

DB_PATH = Path("cell_tracking.db")


def extract_cell_id(ev: Dict[str, Any]) -> str:
    """Safely extracts a clean cell ID string from any event or trajectory payload."""
    for key in ["cell_id", "label_id", "parent_id", "cid", "node"]:
        val = ev.get(key)
        if val is not None:
            # Handle tuple nodes (frame, cid)
            if isinstance(val, tuple) and len(val) >= 2:
                return str(val[1])
            val_str = str(val).strip()
            if val_str.startswith("(") and "," in val_str:
                try:
                    return val_str.strip("()").split(",")[1].strip()
                except Exception:
                    pass
            if val_str not in ["", "N/A", "None", "nan", "NULL", "null"]:
                return val_str
    return "1"


def extract_frame_idx(ev: Dict[str, Any]) -> int:
    """Safely extracts a clean integer frame index from any event or trajectory payload."""
    for key in ["frame", "frame_index", "begin_frame", "t", "frame_idx"]:
        val = ev.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
    node = ev.get("node")
    if isinstance(node, tuple) and len(node) >= 2:
        return int(node[0])
    if isinstance(node, str) and node.startswith("(") and "," in node:
        try:
            return int(node.strip("()").split(",")[0].strip())
        except Exception:
            pass
    return 0


class DatabaseManager:
    """
    SQLite Database Manager for persisting cell tracking experiments and biological analytics.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Creates table schemas and indexes if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Experiments Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    seq_name TEXT NOT NULL,
                    total_frames INTEGER NOT NULL,
                    total_cells INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 2. Cell Events Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cell_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    frame_index INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    cell_id TEXT NOT NULL,
                    details TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
                );
            """)

            # 3. Cell Trajectories / Morphology Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cell_trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    frame_index INTEGER NOT NULL,
                    cell_id INTEGER NOT NULL,
                    centroid_y REAL,
                    centroid_x REAL,
                    area REAL,
                    circularity REAL,
                    eccentricity REAL,
                    speed REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
                );
            """)

            # 4. LLM Biological Reports Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    report_markdown TEXT NOT NULL,
                    biomarkers_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
                );
            """)

            # Indexes for high performance queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_exp ON cell_events(experiment_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_traj_exp ON cell_trajectories(experiment_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_reports_exp ON reports(experiment_id);")

            conn.commit()

    def save_experiment_run(
        self,
        dataset_name: str,
        seq_name: str,
        total_frames: int,
        total_cells: int,
        events: List[Dict[str, Any]],
        frames_payload: List[Dict[str, Any]],
        biomarkers: Optional[Dict[str, Any]] = None,
        report_markdown: Optional[str] = None,
    ) -> int:
        """
        Saves a complete tracking run payload to the SQLite database with normalized fields.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert Experiment
            cursor.execute("""
                INSERT INTO experiments (dataset_name, seq_name, total_frames, total_cells)
                VALUES (?, ?, ?, ?);
            """, (dataset_name, seq_name, total_frames, total_cells))
            exp_id = cursor.lastrowid

            # Insert Cell Events
            for ev in events:
                frame_idx = extract_frame_idx(ev)
                etype = str(ev.get("event_type", "continuity")).lower()
                cid = extract_cell_id(ev)
                
                details = ev.get("details")
                if not details or "N/A" in str(details) or "None" in str(details):
                    if etype == "division":
                        details = f"Parent Cell #{cid} completed mitosis division at Frame {frame_idx}."
                    elif etype == "death":
                        details = f"Cell #{cid} underwent apoptosis / dropout at Frame {frame_idx}."
                    elif etype == "appearance":
                        details = f"New Cell #{cid} entered field of view at Frame {frame_idx}."
                    else:
                        details = f"Cell #{cid} maintained continuous tracking trajectory at Frame {frame_idx}."

                cursor.execute("""
                    INSERT INTO cell_events (experiment_id, frame_index, event_type, cell_id, details)
                    VALUES (?, ?, ?, ?, ?);
                """, (exp_id, frame_idx, etype, cid, details))

            # Insert Trajectories
            for f in frames_payload:
                t = extract_frame_idx(f)
                for cell in f.get("cells", []):
                    cid = extract_cell_id(cell)
                    cy, cx = cell.get("centroid", [0.0, 0.0])
                    area = cell.get("area", 0)
                    cursor.execute("""
                        INSERT INTO cell_trajectories (experiment_id, frame_index, cell_id, centroid_y, centroid_x, area)
                        VALUES (?, ?, ?, ?, ?, ?);
                    """, (exp_id, t, cid, cy, cx, area))

            # Insert LLM Report if available
            if report_markdown:
                bm_json = json.dumps(biomarkers) if biomarkers else None
                cursor.execute("""
                    INSERT INTO reports (experiment_id, report_markdown, biomarkers_json)
                    VALUES (?, ?, ?);
                """, (exp_id, report_markdown, bm_json))

            conn.commit()
            print(f"[Database] Persisted experiment #{exp_id} ({dataset_name} seq {seq_name}, {total_cells} cells) to cell_tracking.db")
            return exp_id

    def get_recent_experiments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns recent tracking experiment records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM experiments ORDER BY id DESC LIMIT ?;
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_experiment_details(self, exp_id: int) -> Dict[str, Any]:
        """Fetches complete normalized data for an experiment ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
            exp = cursor.fetchone()
            if not exp:
                return {}

            cursor.execute("SELECT * FROM cell_events WHERE experiment_id = ? ORDER BY frame_index ASC", (exp_id,))
            events_raw = [dict(r) for r in cursor.fetchall()]
            
            # Normalize event key names for seamless cross-component compatibility
            events = []
            for ev in events_raw:
                c_id = extract_cell_id(ev)
                f_idx = extract_frame_idx(ev)
                ev_normalized = {
                    "id": ev.get("id"),
                    "experiment_id": ev.get("experiment_id"),
                    "frame": f_idx,
                    "frame_index": f_idx,
                    "cell_id": c_id,
                    "label_id": c_id,
                    "event_type": ev.get("event_type", "continuity"),
                    "details": ev.get("details", f"Cell #{c_id} trajectory at Frame {f_idx}."),
                }
                events.append(ev_normalized)

            cursor.execute("SELECT * FROM reports WHERE experiment_id = ? ORDER BY id DESC LIMIT 1", (exp_id,))
            rep = cursor.fetchone()
            report_data = dict(rep) if rep else None

            return {
                "experiment": dict(exp),
                "events": events,
                "report": report_data,
            }

    def maintain_database(self) -> Dict[str, Any]:
        """
        Database maintenance routine:
        - Cleans up corrupted/legacy 'N/A' cell_id records.
        - Deletes orphan records.
        - Runs VACUUM and ANALYZE for SQLite database optimization.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Replace invalid cell_id entries with '1'
            cursor.execute("UPDATE cell_events SET cell_id = '1' WHERE cell_id IN ('N/A', 'None', 'nan', '');")
            
            # Delete orphan cell_events and cell_trajectories
            cursor.execute("DELETE FROM cell_events WHERE experiment_id NOT IN (SELECT id FROM experiments);")
            cursor.execute("DELETE FROM cell_trajectories WHERE experiment_id NOT IN (SELECT id FROM experiments);")
            cursor.execute("DELETE FROM reports WHERE experiment_id NOT IN (SELECT id FROM experiments);")

            conn.commit()

        # Optimize database file
        with self._get_connection() as conn:
            conn.execute("ANALYZE;")
            conn.execute("VACUUM;")

        stats = self.get_database_stats()
        print(f"[Database] Maintenance Complete! Total experiments: {stats['total_experiments']}, events: {stats['total_events']}")
        return stats

    def get_database_stats(self) -> Dict[str, int]:
        """Returns row counts across all database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            n_exp = cursor.execute("SELECT count(*) FROM experiments;").fetchone()[0]
            n_events = cursor.execute("SELECT count(*) FROM cell_events;").fetchone()[0]
            n_traj = cursor.execute("SELECT count(*) FROM cell_trajectories;").fetchone()[0]
            n_rep = cursor.execute("SELECT count(*) FROM reports;").fetchone()[0]
            return {
                "total_experiments": n_exp,
                "total_events": n_events,
                "total_trajectories": n_traj,
                "total_reports": n_rep,
            }


if __name__ == "__main__":
    db = DatabaseManager()
    stats = db.maintain_database()
    print(f"[Database] SQLite Engine Ready! Stats: {stats}")

