import torch

def check_mps_sparse():
    if not torch.backends.mps.is_available():
        print("MPS not available.")
        return

    device = torch.device("mps")
    print(f"Testing MPS on {device}")

    # Create sparse tensor
    i = [[0, 1, 1],
         [2, 0, 2]]
    v =  [3, 4, 5]
    s = torch.sparse_coo_tensor(i, v, (2, 3), dtype=torch.float32, device=device)
    print("Sparse tensor created on MPS.")

    # Create dense tensor
    d = torch.randn(3, 2, device=device)

    # Try sparse mm
    try:
        res = torch.sparse.mm(s, d)
        print("torch.sparse.mm works on MPS!")
        print(res)
    except Exception as e:
        print(f"torch.sparse.mm failed on MPS: {e}")

if __name__ == "__main__":
    check_mps_sparse()
