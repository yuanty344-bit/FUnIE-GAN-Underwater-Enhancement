"""
会话管理

保存/恢复当前工作状态：打开的图像路径、滤镜参数、增强历史。
"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class SessionManager:
    """会话管理器"""

    def __init__(self, session_dir: str = "./temp"):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def save(self, data: Dict[str, Any], name: str = "autosave") -> bool:
        """保存会话"""
        data["saved_at"] = datetime.now().isoformat()
        data["session_name"] = name
        path = self.session_dir / f"{name}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"会话已保存: {path}")
            return True
        except Exception as e:
            logger.error(f"会话保存失败: {e}")
            return False

    def restore(self, name: str = "autosave") -> Optional[Dict[str, Any]]:
        """恢复会话"""
        path = self.session_dir / f"{name}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"会话已恢复: {path}")
            return data
        except Exception as e:
            logger.error(f"会话恢复失败: {e}")
            return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有会话"""
        sessions = []
        for p in sorted(self.session_dir.glob("*.json"), reverse=True):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "name": data.get("session_name", p.stem),
                    "path": str(p),
                    "saved_at": data.get("saved_at", ""),
                    "image": data.get("image_path", ""),
                })
            except Exception:
                pass
        return sessions

    def delete(self, name: str) -> bool:
        """删除会话"""
        path = self.session_dir / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False
