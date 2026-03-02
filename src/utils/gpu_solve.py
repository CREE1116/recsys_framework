"""Backward compatibility: use gpu_accel.py instead."""
from src.utils.gpu_accel import gpu_cholesky_solve, gpu_gram_solve, get_device

__all__ = ['gpu_cholesky_solve', 'gpu_gram_solve', 'get_device']
