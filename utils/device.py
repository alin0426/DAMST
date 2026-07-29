import torch

def setup_device(args):
    try:
        import torch_npu  # noqa: F401
    except Exception:
        torch_npu = None

    if args.device == -1:
        return torch.device("cpu")

    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(args.device)
        return torch.device(f"npu:{args.device}")

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        return torch.device(f"cuda:{args.device}")

    return torch.device("cpu")


def empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "npu":
        torch.npu.empty_cache()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize()
