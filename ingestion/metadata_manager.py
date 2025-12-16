"""
ingestion/metadata_manager.py
功能：项目元数据注册与查询
"""
import sqlite3
import json
from typing import Dict

DB_PATH = "tender_projects.db"

class ProjectRegistry:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._get_conn()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                project_name TEXT PRIMARY KEY,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def register_project(self, project_name: str, metadata: Dict):
        """注册项目到本地 SQLite"""
        # 确保 metadata 可序列化
        if not metadata: metadata = {}

        # 简单清洗
        if "type" in metadata:
            if "水库" in metadata["type"]: metadata["type"] = "水库"

        meta_json = json.dumps(metadata, ensure_ascii=False)

        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO projects (project_name, metadata_json) VALUES (?, ?)",
                (project_name, meta_json)
            )
            conn.commit()
            print(f"✅ [DB] 项目 '{project_name}' 元数据已保存。")
        except Exception as e:
            print(f"❌ [DB] 注册失败: {e}")
        finally:
            conn.close()

    def get_metadata(self, project_name: str) -> Dict:
        """获取元数据"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT metadata_json FROM projects WHERE project_name = ?", (project_name,))
        row = cursor.fetchone()
        conn.close()

        if row:
            try:
                base_meta = json.loads(row[0])
                base_meta["project_name"] = project_name
                return base_meta
            except:
                pass
        return {"project_name": project_name}
