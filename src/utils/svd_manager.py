"""Backward compatibility: SVDCacheManager is now in gpu_accel.py"""
from src.utils.gpu_accel import SVDCacheManager, gpu_cholesky_solve, gpu_gram_solve, get_device

__all__ = ['SVDCacheManager', 'gpu_cholesky_solve', 'gpu_gram_solve', 'get_device']