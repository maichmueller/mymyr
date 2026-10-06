"""The counter-based RNG of mymyr (rl/rng.hpp: Philox-4x32-10) in jnp, for jitted code.

``philox(counters, seed)`` equals ``mymyr.rl.torch.rng.philox`` / ``rl::rng::philox4x32_10`` bit for bit; the draws of
the environments and of :func:`mymyr.rl.her_relabel` are Philox blocks of (seed; draw, env, purpose). JAX has no
uint64 without x64, so 64-bit values travel as uint32 (low, high) pairs and the 32 x 32 -> 64-bit products are
composed from 16-bit halves.
"""

import jax
import jax.numpy as jnp

__all__ = ["mulhilo32", "philox", "block", "bits64", "below", "successor_index", "PURPOSE_SUCCESSOR", "PURPOSE_HER"]

PURPOSE_SUCCESSOR = 0  # rl::rng::k_successor
PURPOSE_HER = 1  # rl::rng::k_her (the future step and the goal atoms of a relabelled transition)

_M0, _M1 = 0xD2511F53, 0xCD9E8D57
_W0, _W1 = 0x9E3779B9, 0xBB67AE85


def mulhilo32(a, b):
    """(high, low) 32-bit words of the 64-bit product of uint32 a and b."""
    a = jnp.asarray(a, jnp.uint32)
    b = jnp.asarray(b, jnp.uint32)
    lo = a * b
    a0, a1 = a & 0xFFFF, a >> 16
    b0, b1 = b & 0xFFFF, b >> 16
    p00 = a0 * b0
    p01 = a0 * b1
    p10 = a1 * b0
    p11 = a1 * b1
    mid = (p00 >> 16) + (p01 & 0xFFFF) + (p10 & 0xFFFF)
    hi = p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16)
    return hi, lo


def philox(c0, c1, c2, c3, k0, k1):
    """Philox-4x32-10 of the counter (c0, c1, c2, c3) under the key (k0, k1): four uint32 arrays (broadcast)."""
    c = jnp.broadcast_arrays(*(jnp.asarray(x, jnp.uint32) for x in (c0, c1, c2, c3, k0, k1)))

    def round_(_, v):
        c0, c1, c2, c3, k0, k1 = v
        hi0, lo0 = mulhilo32(jnp.uint32(_M0), c0)
        hi1, lo1 = mulhilo32(jnp.uint32(_M1), c2)
        return (hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0, k0 + jnp.uint32(_W0), k1 + jnp.uint32(_W1))

    # a loop, not ten unrolled rounds: XLA's CPU backend compiles the unrolled graph slowly (12-16 s per shape)
    c0, c1, c2, c3, _, _ = jax.lax.fori_loop(0, 10, round_, tuple(c))
    return c0, c1, c2, c3


def block(seed, env, draw, purpose):
    """The Philox block of (seed; draw, env, purpose) (rl::rng::block): seed and draw as uint32 [..., 2] (low, high)
    pairs, env and purpose uint32."""
    seed = jnp.asarray(seed, jnp.uint32)
    draw = jnp.asarray(draw, jnp.uint32)
    return philox(draw[..., 0], draw[..., 1], env, purpose, seed[..., 0], seed[..., 1])


def bits64(seed, env, draw, purpose):
    """The first two words of the block: 64 random bits as uint32 (low, high)."""
    b = block(seed, env, draw, purpose)
    return b[0], b[1]


def below(lo, hi, n):
    """rl::rng::below: the high 32 bits of the 64-bit (hi, lo) value times n (uint32 n >= 1): an index in [0, n)."""
    n = jnp.asarray(n, jnp.uint32)
    h_hi, h_lo = mulhilo32(hi, n)  # (x >> 32) * n
    l_hi, _ = mulhilo32(lo, n)  # ((x & 0xFFFFFFFF) * n) >> 32
    s_lo = h_lo + l_hi
    carry = (s_lo < h_lo).astype(jnp.uint32)
    return h_hi + carry


def successor_index(seed, env, draw, count):
    """The random policy's successor index (rl::rng::successor_index) for counts >= 1."""
    lo, hi = bits64(seed, env, draw, PURPOSE_SUCCESSOR)
    return below(lo, hi, count)
