import torch
import numpy as np
import os
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

class SVDCacheManager:
    """
    SVD 결과를 캐싱하고 MPS 가속을 제공하는 관리자 클래스
    """
    def __init__(self, cache_dir='data_cache', device='auto'):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # 디바이스 자동 설정: MPS > CPU
        if device == 'auto':
            self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"[SVD-Manager] Using device: {self.device}")

    @staticmethod
    def get_analysis_dir(config):
        """
        config 정보를 바탕으로 분석 결과가 저장될 디렉토리 경로를 생성합니다.
        """
        dataset_name = config.get('dataset_name', 'unknown')
        model_name = config['model']['name']
        run_name = config.get('run_name', 'default')
        
        if run_name and run_name != 'default':
            folder = f"{model_name}__{run_name}"
        else:
            folder = model_name
            
        base_path = os.path.join('trained_model', dataset_name, folder, 'analysis')
        return base_path

    def get_svd(self, X_sparse, k, dataset_name=None, force_recompute=False):
        """
        캐시에서 SVD 결과를 가져오거나 새로 계산합니다.
        
        Args:
            X_sparse: Scipy sparse matrix (CSR 형식 권장)
            k: SVD rank
            dataset_name: 캐시 파일명에 사용될 데이터셋 이름
            force_recompute: True면 캐시를 무시하고 재계산
        
        Returns:
            u, s, v, total_energy
        """
        # 입력 검증 및 k 캡핑
        M, N = X_sparse.shape
        if k >= min(M, N):
            print(f"[SVD-Manager] Warning: k({k}) is too large for matrix {M}x{N}. Capping to {min(M, N) - 1}")
            k = min(M, N) - 1
            if k < 1: k = 1
        
        # 캐시 확인
        if dataset_name and not force_recompute:
            cache_path = os.path.join(self.cache_dir, f"svd_{dataset_name}_k{k}.pt")
            if os.path.exists(cache_path):
                print(f"[SVD-Manager] 캐시에서 로딩: {dataset_name} (k={k})...")
                checkpoint = torch.load(cache_path, map_location='cpu')
                
                # Handle cases where total_energy might be missing from cache
                u, s, v = checkpoint['u'], checkpoint['s'], checkpoint['v']
                total_energy = checkpoint.get('total_energy')
                if total_energy is None:
                    print("[SVD-Manager] Warning: 'total_energy' missing from cache. Recalculating...")
                    total_energy = float(np.sum(X_sparse.data ** 2))
                
                return u, s, v, total_energy

        print(f"[SVD-Manager] SVD 계산 시작 (k={k}, shape={X_sparse.shape})...")
        start_time = time.time()
        
        # 1. Total Energy 계산 (Frobenius norm squared)
        total_energy = float(np.sum(X_sparse.data ** 2))
        
        # 2. SVD 계산
        min_dim = min(M, N)
        if self.device == 'mps' and min_dim >= 5000:
             u, s, v = self.mps_randomized_svd(X_sparse, k)
        else:
             if self.device == 'mps':
                 print(f"[SVD-Manager] Matrix too small ({min_dim} < 5000) or incompatible for Randomized SVD. Using CPU SVD.")
             u, s, v = self.cpu_svd(X_sparse, k)

        elapsed = time.time() - start_time
        print(f"[SVD-Manager] SVD 완료 ({elapsed:.2f}초)")

        # 3. 캐시 저장
        if dataset_name:
            cache_path = os.path.join(self.cache_dir, f"svd_{dataset_name}_k{k}.pt")
            print(f"[SVD-Manager] 결과 저장 중: {cache_path}...")
            # Save to cache (as tensors), including total_energy
            torch.save({'u': u, 's': s, 'v': v, 'total_energy': total_energy}, cache_path)
            
        return u, s, v, total_energy

    def cpu_svd(self, X_sparse, k):
        """
        CPU에서 scipy를 이용한 SVD 계산
        Small matrix: Dense SVD (Faster for large k)
        Large matrix: Sparse Iterative SVDS
        """
        min_dim = min(X_sparse.shape)
        
        # Optimization: If matrix is small, use Dense SVD
        # svds is very slow when k is close to min_dim (full rank)
        if min_dim < 2000:
            print(f"[CPU-SVD] Matrix small ({X_sparse.shape}), using Dense SVD (scipy.linalg.svd)...")
            from scipy.linalg import svd
            X_dense = X_sparse.toarray()
            # full_matrices=False -> u(M,K), s(K,), vt(K,N)
            u, s, vt = svd(X_dense, full_matrices=False)
            
            # svd returns sorted s by default
            # Top-k truncation
            u = u[:, :k]
            s = s[:k]
            vt = vt[:k, :]
            
        else:
            print(f"[CPU-SVD] Matrix large, using Sparse Iterative SVDS (scipy.sparse.linalg.svds)...")
            # scipy svds returns: u (M,k), s (k,), vt (k,N)
            # We want U, S, V where X ~ U @ S @ V.T
            # svds returns u, s, vt. U=u, S=s, V=vt.T
            u, s, vt = svds(X_sparse, k=k)
            
            # svds는 특이값을 오름차순으로 반환하므로 내림차순으로 정렬
            idx = np.argsort(s)[::-1]
            s = s[idx]
            u = u[:, idx]
            vt = vt[idx, :] # vt is (k, N)
        
        # Convert to Tensor (Clean return for caller)
        u = torch.from_numpy(u.copy()).float()
        s = torch.from_numpy(s.copy()).float()
        v = torch.from_numpy(vt.T.copy()).float() # V should be (N, k)
        
        return u, s, v

    def _orthonormalize(self, Y, device):
        """
        Orthonormalize columns of Y using Eigen decomposition (CPU fallback for stability).
        Q = Y @ V @ S^-0.5
        """
        if Y.shape[1] == 0:
            return Y
            
        # Gram matrix C = Y^T @ Y (q x q)
        C = torch.mm(Y.t(), Y)
        
        try:
            # Move to CPU for stable/available Eigen decomposition
            C_cpu = C.cpu()
            S, V = torch.linalg.eigh(C_cpu)
            
            # Take only positive eigenvalues and apply inverse square root
            S_inv_sqrt = torch.where(S > 1e-12, 1.0 / torch.sqrt(S), torch.zeros_like(S))
            
            # Move results back to device
            V = V.to(device)
            S_inv_sqrt = S_inv_sqrt.to(device)
            
            # Q = Y @ (V @ diag(S_inv_sqrt))
            Q = torch.mm(Y, V * S_inv_sqrt.unsqueeze(0))
            return Q
        except Exception as e:
            print(f"[SVD-Manager] Warning: Orthonormalization failed ({e}). Fallback to simple scaling.")
            return Y / Y.norm(dim=0, keepdim=True).clamp(min=1e-12)

    @torch.no_grad()
    def mps_randomized_svd(self, X_sparse, k, n_iter=2, oversampling=10, batch_size=2000):
        """
        MPS 가속을 이용한 Randomized SVD (Halko et al., 2011)
        """
        device = torch.device("mps")
        M, N = X_sparse.shape
        q = min(k + oversampling, M, N)
        if q < 1: q = 1
        
        print(f"[MPS-SVD] 파라미터: k={k}, q={q}, n_iter={n_iter}, batch_size={batch_size}")
        
        # Phase 1: Sketching with Power Iteration
        print(f"[MPS-SVD] Phase 1: Sketching (랜덤 투영)...")
        G = torch.randn(N, q, device=device, dtype=torch.float32)
        Y = self._sparse_mm_batched(X_sparse, G, batch_size, device)
        Q = self._orthonormalize(Y, device)
        
        # Power iteration으로 정확도 향상
        for iter_idx in range(n_iter):
            print(f"[MPS-SVD] Power iteration {iter_idx + 1}/{n_iter}...")
            # Z = A^T @ Q
            Z = self._sparse_mm_transposed_batched(X_sparse, Q, batch_size, device)
            Q_z = self._orthonormalize(Z, device)
            # Y = A @ Q_z
            Y = self._sparse_mm_batched(X_sparse, Q_z, batch_size, device)
            Q = self._orthonormalize(Y, device)
            del Z, Q_z
        
        # Phase 2: Basis Q is already orthonormalized in Phase 1 loop
        
        # Phase 3: Projection (B = Q^T @ A)
        # Phase 3: Projection (B = Q^T @ A)
        print(f"[MPS-SVD] Phase 3: 투영 행렬 계산...")
        B = self._sparse_mm_transposed_batched(X_sparse, Q, batch_size, device)
        B = B.t() # (q x N)
        
        # Phase 4: Small SVD (MPS Optimized via Spectral Trick)
        print(f"[MPS-SVD] Phase 4: 작은 행렬 SVD 최적화 (q x q SVD)...")
        # C = B @ B^T (q x q)
        C = torch.mm(B, B.t())
        C_cpu = C.cpu()
        
        try:
            # Symmetric eigen decomposition (eigh is faster/stable for symmetric)
            S2_cpu, U_hat_cpu = torch.linalg.eigh(C_cpu)
            
            # Descending order
            idx = torch.argsort(S2_cpu, descending=True)
            S2_cpu = S2_cpu[idx]
            U_hat_cpu = U_hat_cpu[:, idx]
            
            # Singular values
            S_cpu = torch.sqrt(torch.clamp(S2_cpu, min=0.0))
        except Exception as e:
            print(f"[MPS-SVD] SVD failed ({e}), falling back to svd...")
            U_hat_cpu, S_cpu, _ = torch.linalg.svd(C_cpu, full_matrices=False)

        U_hat = U_hat_cpu.to(device)
        S = S_cpu.to(device)
        
        # Recover V = B^T @ U_hat @ S^-1  (Note: B was transposed above, so B.t() is N x q)
        S_inv = torch.where(S > 1e-12, 1.0 / S, torch.zeros_like(S))
        V = torch.mm(B.t(), U_hat) * S_inv
        
        # Final reconstruction (U = Q @ U_hat)
        U = torch.mm(Q, U_hat)
        
        # Top-k 추출
        U = U[:, :k].cpu()
        S = S[:k].cpu()
        V = V[:, :k].cpu()
        
        print(f"[MPS-SVD] 완료! 특이값 범위: [{S[-1]:.4f}, {S[0]:.4f}]")
        return U, S, V
    @staticmethod
    def _sparse_mm_batched(X_sparse, Y_dense, batch_size, device):
        """
        배치로 나누어 X_sparse @ Y_dense 계산 (메모리 효율)
        X_sparse: (M x N) scipy sparse
        Y_dense: (N x q) torch tensor on device
        Returns: (M x q) torch tensor on device
        """
        M = X_sparse.shape[0]
        q = Y_dense.shape[1]
        result = torch.zeros(M, q, device=device)
        
        # Y_dense는 이미 device에 있다고 가정 (또는 device로 이동)
        if Y_dense.device.type != device:
             Y_dense = Y_dense.to(device)

        for i in range(0, M, batch_size):
            end = min(i + batch_size, M)
            # Sparse matrix slicing and dense conversion
            # X_sparse는 CPU에 있으므로 해당 배치만 GPU로 이동
            A_batch = torch.from_numpy(
                X_sparse[i:end].toarray()
            ).float().to(device)
            
            # Y_dense는 이미 GPU에 있으므로 바로 연산
            result[i:end] = torch.mm(A_batch, Y_dense)
            
            del A_batch
        
        return result
    
    @staticmethod
    def _sparse_mm_transposed_batched(X_sparse, Y_dense, batch_size, device):
        """
        배치로 나누어 X_sparse^T @ Y_dense 계산
        X_sparse: (M x N) scipy sparse
        Y_dense: (M x q) torch tensor on device
        Returns: (N x q) torch tensor on device
        """
        M, N = X_sparse.shape
        q = Y_dense.shape[1]
        result = torch.zeros(N, q, device=device)
        
        if Y_dense.device.type != device:
             Y_dense = Y_dense.to(device)
        
        for i in range(0, M, batch_size):
            end = min(i + batch_size, M)
            A_batch = torch.from_numpy(
                X_sparse[i:end].toarray()
            ).float().to(device)
            
            # Y_batch slicing
            Y_batch = Y_dense[i:end]
            
            result += torch.mm(A_batch.t(), Y_batch)
            
            del A_batch # Y_batch는 view일 수 있으므로 del 불필요하지만 명시적 해제
        
        return result

    def clear_cache(self, dataset_name=None):
        """캐시 삭제"""
        if dataset_name:
            pattern = f"svd_{dataset_name}_*.pt"
            import glob
            files = glob.glob(os.path.join(self.cache_dir, pattern))
            for f in files:
                os.remove(f)
                print(f"[SVD-Manager] 삭제: {f}")
        else:
            import shutil
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
            print(f"[SVD-Manager] 전체 캐시 삭제 완료")


# 사용 예시
if __name__ == "__main__":
    from scipy.sparse import random
    
    # 테스트 데이터 생성
    X = random(10000, 5000, density=0.01, format='csr')
    
    # SVD Manager 초기화 (MPS 자동 감지)
    manager = SVDCacheManager(device='auto')
    
    # SVD 계산 (캐싱 적용)
    u, s, v, energy = manager.get_svd(X, k=50, dataset_name='test_data')
    
    print(f"\n결과:")
    print(f"  U shape: {u.shape}")
    print(f"  S shape: {s.shape}")
    print(f"  V shape: {v.shape}")
    print(f"  Total energy: {energy:.2f}")
    print(f"  Top-5 특이값: {s[:5]}")