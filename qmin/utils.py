import torch

device = "cpu"

# use_direct = True


def use_cuda() -> bool:
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report_device()
    return device == "cuda"


def use_cpu() -> bool:
    global device
    device = "cpu"
    report_device()
    return True


def used_device() -> str:
    return device


def report_device() -> None:
    print(f"Using device: {device}")
