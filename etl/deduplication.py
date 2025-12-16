"""
etl/deduplication.py
功能：文件指纹计算与查重。
"""
import hashlib
import os
import sqlite3

class DeduplicationService:
    def __init__(self, db_path="tender_projects.db"):
        self.db_path = db_path
        self._init_table()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_table(self):
        conn = self._get_conn()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS processed_files (
                file_hash TEXT PRIMARY KEY,
                file_name TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def get_file_hash(self, file_path: str) -> str:
        """计算文件的 MD5 哈希值"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def is_processed(self, file_path: str) -> bool:
        """检查文件是否处理过"""
        if not os.path.exists(file_path): return False
        try:
            file_hash = self.get_file_hash(file_path)
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM processed_files WHERE file_hash = ?", (file_hash,))
            exists = cursor.fetchone() is not None
            conn.close()
            return exists
        except Exception as e:
            print(f"⚠️ 查重失败: {e}，默认未处理")
            return False

    def mark_as_processed(self, file_path: str):
        """标记文件为已处理"""
        try:
            file_hash = self.get_file_hash(file_path)
            file_name = os.path.basename(file_path)
            conn = self._get_conn()
            conn.execute(
                "INSERT OR IGNORE INTO processed_files (file_hash, file_name) VALUES (?, ?)",
                (file_hash, file_name)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️ 标记处理失败: {e}")
