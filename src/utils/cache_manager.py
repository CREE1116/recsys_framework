"""
Framework Cache Manager
=======================
캐시 라이프사이클을 관리하는 추상 인터페이스 및 레지스트리.

GlobalCacheManager: 데이터셋 단위 공유 캐시. 모델 간 공유. (예: SVD, MNAR Gamma, SLIM Matrix, GramEigen)
"""

from abc import ABC, abstractmethod


class CacheManager(ABC):
    """캐시 매니저 추상 인터페이스"""

    @abstractmethod
    def summary(self) -> dict:
        """현재 캐시 상태 요약 (로깅용)"""
        pass

    @abstractmethod
    def invalidate(self, key: str = None):
        """특정 키 또는 전체 캐시 무효화.
        key=None이면 전체 무효화."""
        pass


class GlobalCacheManager(CacheManager):
    """
    데이터셋 단위 공유 캐시. 여러 모델이 같은 캐시를 참조.
    - 데이터셋이 변경될 때만 무효화 대상.
    - 예: SVDCacheManager, MNARGammaCacheManager
    """
    pass


class CacheRegistry:
    """
    Trainer가 사용하는 캐시 매니저 레지스트리.
    모델이 register()로 등록하면, Trainer가 자동으로 라이프사이클을 관리.
    """
    def __init__(self):
        self._managers: dict[str, CacheManager] = {}

    def register(self, name: str, manager: CacheManager):
        self._managers[name] = manager

    def get(self, name: str) -> CacheManager:
        return self._managers.get(name)

    def all(self) -> dict[str, CacheManager]:
        return dict(self._managers)

    def invalidate_all(self):
        """전체 캐시 무효화 (데이터셋 변경 시)"""
        for mgr in self._managers.values():
            mgr.invalidate()

    def log_status(self):
        """Print aggregated cache summary: entry count and total size."""
        if not self._managers:
            return
        total_entries = 0
        total_mb = 0.0
        for mgr in self._managers.values():
            s = mgr.summary()
            total_entries += s.get('files', 0) + s.get('entries', 0)
            total_mb += s.get('size_mb', 0)
        print(f"[Cache] {total_entries} entries, ~{total_mb:.0f} MB on disk")
