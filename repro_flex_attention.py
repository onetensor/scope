import argparse
import os

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return default if v is None else int(v)


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean env var {key}={v!r}")


def build_sliding_window_blockmask(*, T: int, H: int, block_size: int, window_blocks: int) -> BlockMask:
    assert T % block_size == 0
    num_blocks = T // block_size
    device = torch.device("cuda")
    window_blocks = min(int(window_blocks), int(num_blocks))

    def token_causal(b, h, q_idx, kv_idx):
        q = q_idx.to(torch.long)
        k = kv_idx.to(torch.long)
        return q >= k

    q_blocks = torch.arange(num_blocks, device=device, dtype=torch.int32)  # [QB]
    offs = torch.arange(window_blocks, device=device, dtype=torch.int32)  # [W]
    kv_blocks = (q_blocks[:, None] - offs[None, :]).clamp(min=0)  # [QB,W]
    # Match FlexAttention's expected kv_len=T by making the index list length == num_blocks.
    kv_indices = torch.zeros((num_blocks, num_blocks), device=device, dtype=torch.int32)  # [QB,QB]
    kv_indices[:, :window_blocks] = kv_blocks
    kv_num_blocks = torch.minimum(q_blocks + 1, q_blocks.new_full((num_blocks,), window_blocks))  # [QB]

    kv_indices = kv_indices[None, None].expand(1, H, num_blocks, num_blocks).contiguous()  # [1,H,QB,QB]
    kv_num_blocks = kv_num_blocks[None, None].expand(1, H, num_blocks).contiguous()  # [1,H,QB]
    zeros_num = torch.zeros_like(kv_num_blocks)
    zeros_idx = torch.zeros_like(kv_indices)

    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        zeros_num,
        zeros_idx,
        BLOCK_SIZE=block_size,
        mask_mod=token_causal,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Repro harness for FlexAttention compilation failures.")
    parser.add_argument("--T", type=int, default=_env_int("T", 65536))
    parser.add_argument("--H", type=int, default=_env_int("H", 8))
    parser.add_argument("--D", type=int, default=_env_int("D", 144))
    parser.add_argument("--block-size", type=int, default=_env_int("BLOCK_SIZE", 128))
    parser.add_argument("--window-blocks", type=int, default=_env_int("WINDOW_BLOCKS", 14))
    parser.add_argument("--scale", type=float, default=float(os.environ.get("SCALE", "0.12")))
    parser.add_argument("--compile", action="store_true", default=_env_bool("TORCH_COMPILE", False))
    args = parser.parse_args()

    assert torch.cuda.is_available(), "requires CUDA"
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.manual_seed(0)

    q = torch.randn((1, args.H, args.T, args.D), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn((1, args.H, args.T, args.D), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn((1, args.H, args.T, args.D), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bm = build_sliding_window_blockmask(T=args.T, H=args.H, block_size=args.block_size, window_blocks=args.window_blocks)

    def fn(q, k, v):
        out = flex_attention(q, k, v, block_mask=bm, scale=args.scale)
        return out.sum()

    if args.compile:
        fn = torch.compile(fn)  # type: ignore[assignment]

    loss = fn(q, k, v)
    loss.backward()
    torch.cuda.synchronize()
    print(
        f"ok: T={args.T} H={args.H} D={args.D} block={args.block_size} window_blocks={args.window_blocks} compile={args.compile}"
    )


if __name__ == "__main__":
    main()

