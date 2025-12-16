"""
=== Python代码文件: file_manager.py (V2.0 级联筛选增强版 - 完整代码) ===
"""
import os
import shutil
import json
from typing import List, Dict, Optional


class FileManager:
    """
    负责管理 data_repository 中的物理文件和元数据。
    元数据存储在 metadata_registry.json 中，用于支持高级筛选。
    """

    def __init__(self, base_dir="data_repository"):
        self.base_dir = base_dir
        # 元数据文件的绝对路径
        self.metadata_path = os.path.join(self.base_dir, "metadata_registry.json")
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    # --- 私有辅助方法 ---
    def _load_metadata(self) -> Dict:
        """安全地加载元数据文件，如果文件不存在或损坏则返回空字典。"""
        if not os.path.exists(self.metadata_path):
            return {}
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # 如果文件损坏或读取错误，返回空字典以避免程序崩溃
            return {}

    def _save_metadata(self, data: Dict):
        """将元数据以格式化的JSON形式保存。"""
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # --- 核心功能方法 ---

    def register_file_metadata(self, project: str, tag: str, filename: str):
        """
        [ETL接口] 注册或更新一个文件的元数据。
        这个函数应该在文件被成功处理并入库到向量数据库后，由ETL pipeline调用。
        """
        if not all([project, tag, filename]):
            print("[FileManager] Warning: 元数据注册失败，项目、标签或文件名为空。")
            return

        metadata = self._load_metadata()
        if project not in metadata:
            metadata[project] = {}

        metadata[project][filename] = {"tag": tag}
        self._save_metadata(metadata)
        print(f"[FileManager] 元数据已注册: Project='{project}', File='{filename}', Tag='{tag}'")

    def get_folders(self) -> List[str]:
        """[旧功能] 获取所有项目文件夹名称。"""
        try:
            items = os.listdir(self.base_dir)
            folders = [item for item in items if os.path.isdir(os.path.join(self.base_dir, item))]
            return sorted(folders)
        except Exception:
            return []

    def create_folder(self, folder_name: str) -> bool:
        """[旧功能] 创建新项目文件夹，并进行名称安全过滤。"""
        safe_name = "".join([c for c in folder_name if c.isalnum() or c in (' ', '_', '-')]).strip()
        if not safe_name:
            return False

        target_path = os.path.join(self.base_dir, safe_name)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
            return True
        return False

    def save_file(self, uploaded_file, folder_name: str) -> Optional[str]:
        """[旧功能] 保存上传的文件到指定文件夹，返回文件的绝对路径。"""
        try:
            target_dir = os.path.join(self.base_dir, folder_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            file_path = os.path.join(target_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return os.path.abspath(file_path)
        except Exception as e:
            print(f"[FileManager] Error saving file: {e}")
            return None

    def get_all_files(self) -> Dict[str, List[str]]:
        """[旧功能] 获取物理目录树，用于知识库管理页面的展示。"""
        tree = {}
        folders = self.get_folders()
        for folder in folders:
            folder_path = os.path.join(self.base_dir, folder)
            try:
                files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                tree[folder] = sorted(files)
            except OSError:
                continue
        return tree

    def delete_file(self, folder_name: str, filename: str) -> bool:
        """[增强功能] 删除指定文件夹下的物理文件，并同步删除其元数据记录。"""
        file_path = os.path.join(self.base_dir, folder_name, filename)

        # 1. 删除物理文件
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"[FileManager] 物理文件删除失败: {e}")
                return False

        # 2. 同步删除元数据
        metadata = self._load_metadata()
        if folder_name in metadata and filename in metadata.get(folder_name, {}):
            del metadata[folder_name][filename]
            # 如果项目下没有文件了，可以一并删除该项目在元数据中的条目
            if not metadata[folder_name]:
                del metadata[folder_name]
            self._save_metadata(metadata)
            print(f"[FileManager] 元数据已删除: Project='{folder_name}', File='{filename}'")

        return True

    # ==========================================================
    # 🌟 [新功能] 级联筛选核心函数
    # ==========================================================

    def get_tags_for_project(self, project_name: str) -> List[str]:
        """获取指定项目下的所有唯一文档类型（tags）。"""
        metadata = self._load_metadata()
        project_data = metadata.get(project_name, {})
        tags = set(file_info["tag"] for file_info in project_data.values() if "tag" in file_info)
        return sorted(list(tags))

    def get_files_for_project_and_tag(self, project_name: str, tag_name: str) -> List[str]:
        """获取指定项目和指定文档类型下的所有文件名。"""
        metadata = self._load_metadata()
        project_data = metadata.get(project_name, {})
        files = [
            filename for filename, file_info in project_data.items()
            if file_info.get("tag") == tag_name
        ]
        return sorted(files)

