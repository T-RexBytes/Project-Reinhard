"""
app/decoding/reed_solomon.py
-----------------------------
Reed-Solomon FEC over GF(256): classic RS(255, 223) and RS(255, 239).

A Reed-Solomon code treats K data symbols (each a GF(256) element) plus 2*t
parity symbols, forming a codeword of length N = 255. It corrects up to
`t = (N - K) / 2` symbol errors.

  - RS(255, 223) : t = 16  (NIST / CCSDS-style, robust)
  - RS(255, 239) : t = 8   (faster, lighter overhead)

Convention
----------
All coefficient arrays are stored *little-endian*: ``poly[i]`` is the field
coefficient of ``x**i``. A codeword is an ``n``-length array

    cw = [ rem[0..nsym-1] | msg[0..k-1] ]

where the ``k`` message symbols occupy the highest ``k`` powers of ``x`` and the
``nsym = n-k`` parity symbols (remainder of ``msg(x)*x**nsym`` modulo the
generator) occupy the lowest powers. Valid codewords therefore vanish at the
generator roots ``x = alpha**(fcr+i)``.

Algorithm
---------
  syndromes -> Berlekamp-Massey (error locator) -> Chien search (root positions)
  -> Forney (error magnitudes)

Self-contained NumPy implementation (primitive polynomial 0x11d, alpha = 2),
no third-party dependency.

Scope (constrained candidate philosophy): only the two standard RS profiles are
exposed. Shortened / arbitrary-generator RS discovery is out of scope (P2).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

_GF_POLY = 0x11D
_GEN = 2        # primitive element alpha

# ── GF(256) engine (log / antilog tables) ───────────────────────────────────

def _gf_tables():
    log = np.zeros(256, dtype=np.int64)
    exp = np.zeros(255, dtype=np.int64)
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= _GF_POLY
    log[0] = 0
    exp2 = np.concatenate([exp, exp])
    return log, exp2


_LOG, _EXP = _gf_tables()


def gf_mul(a: int, b: int) -> int:
    a, b = a & 0xFF, b & 0xFF
    if a == 0 or b == 0:
        return 0
    return int(_EXP[_LOG[a] + _LOG[b]])


def gf_div(a: int, b: int) -> int:
    b = b & 0xFF
    if b == 0:
        raise ZeroDivisionError("Division by 0 in GF(256).")
    a = a & 0xFF
    if a == 0:
        return 0
    return int(_EXP[(_LOG[a] + 255 - _LOG[b]) % 255])


def gf_pow(a: int, n: int) -> int:
    a = a & 0xFF
    if a == 0:
        return 0 if n > 0 else 1
    if n == 0:
        return 1
    return int(_EXP[(_LOG[a] * n) % 255])


def gf_inv(a: int) -> int:
    return gf_pow(a, 254)


def gf_add(a: int, b: int) -> int:
    return (a & 0xFF) ^ (b & 0xFF)


# ── GF(256) polynomials (little-endian coefficient arrays) ─────────────────

def _trim(p) -> np.ndarray:
    p = list(np.asarray(p, dtype=np.int64))
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return np.array(p, dtype=np.int64)


def gf_poly_mul(p, q) -> np.ndarray:
    p = np.asarray(p, dtype=np.int64)
    q = np.asarray(q, dtype=np.int64)
    if len(p) == 0 or len(q) == 0:
        return np.zeros(1, dtype=np.int64)
    result = np.zeros(len(p) + len(q) - 1, dtype=np.int64)
    for i, ai in enumerate(p):
        a = int(ai) & 0xFF
        if a == 0:
            continue
        la = _LOG[a]
        for j, bj in enumerate(q):
            b = int(bj) & 0xFF
            if b == 0:
                continue
            result[i + j] ^= _EXP[la + _LOG[b]]
    return _trim(result)


def gf_poly_eval(p, x: int) -> int:
    """Horner evaluation at field element x."""
    x = x & 0xFF
    acc = 0
    for c in reversed(np.asarray(p, dtype=np.int64)):
        acc = gf_mul(acc, x) ^ (int(c) & 0xFF)
    return acc & 0xFF


def gf_poly_div(dividend, divisor):
    """Long division (ascending powers). Returns (quotient, remainder).

    Remainder degree < deg(divisor). Divisor must be monic (or have invertible
    leading coefficient).
    """
    dividend = _trim(dividend)
    divisor = _trim(divisor)
    if len(divisor) == 0 or (len(divisor) == 1 and divisor[0] == 0):
        raise ZeroDivisionError("Cannot divide by the zero polynomial.")
    r = list(dividend)
    d = list(divisor)
    dl = len(d)
    lc_inv = gf_inv(d[-1])
    while len(r) >= dl and any(r):
        deg_r = len(r) - 1
        deg_d = dl - 1
        if deg_r < deg_d:
            break
        coef = gf_mul(r[-1], lc_inv)
        shift = deg_r - deg_d
        new_r = list(r)
        for j in range(dl):
            if d[j] == 0:
                continue
            new_r[j + shift] = gf_add(new_r[j + shift], gf_mul(int(d[j]), coef))
        r = _trim(new_r)
    return np.array([0], dtype=np.int64), np.array(r, dtype=np.int64)


# ── RS profile and generator ────────────────────────────────────────────────

@dataclass
class RSParams:
    """One Reed-Solomon profile."""
    name: str
    n: int = 255
    k: int = 223
    fcr: int = 0          # first consecutive root

    @property
    def t(self) -> int:
        return (self.n - self.k) // 2

    @property
    def nsym(self) -> int:
        return self.n - self.k

    def to_dict(self) -> dict:
        return {"name": self.name, "n": self.n, "k": self.k, "t": self.t}


RS255_223 = RSParams(name="rs255_223", n=255, k=223)
RS255_239 = RSParams(name="rs255_239", n=255, k=239)


def _rs_generator_poly(nsym: int, fcr: int = 0) -> np.ndarray:
    """g(x) = prod_{i=fcr}^{fcr+nsym-1} (x - alpha^i), little-endian coeffs."""
    g = np.array([1], dtype=np.int64)
    for i in range(fcr, fcr + nsym):
        g = gf_poly_mul(g, np.array([gf_pow(_GEN, i % 255), 1], dtype=np.int64))
    return g


_gen_cache: dict[tuple[int, int], np.ndarray] = {}


def _generator(params: RSParams) -> np.ndarray:
    key = (params.nsym, params.fcr)
    if key not in _gen_cache:
        _gen_cache[key] = _rs_generator_poly(params.nsym, params.fcr)
    return _gen_cache[key]


# ── Encoder ─────────────────────────────────────────────────────────────────

def rs_encode(msg: np.ndarray, params: RSParams) -> np.ndarray:
    """Systematically encode `params.k` GF(256) symbols into n-length codeword.

    Returns ``[rem (nsym low-power parity) | msg (k high-power message)]``,
    little-endian coefficient order (index i = coefficient of x**i).
    """
    msg = np.asarray(msg, dtype=np.int64).ravel()
    if msg.size != params.k:
        raise ValueError(f"Message length {msg.size} != RS k={params.k}")
    nsym = params.nsym
    gen = _generator(params)
    shifted = np.concatenate([np.zeros(nsym, dtype=np.int64), msg])
    _, rem = gf_poly_div(shifted, gen)
    if len(rem) < nsym:
        rem = np.concatenate([rem, np.zeros(nsym - len(rem), dtype=np.int64)])
    codeword = np.zeros(params.n, dtype=np.int64)
    codeword[:len(rem)] = rem
    codeword[nsym:] = msg
    return codeword


# ── Decoder ─────────────────────────────────────────────────────────────────

def _syndromes(cw, params: RSParams) -> np.ndarray:
    nsym, fcr = params.nsym, params.fcr
    S = np.zeros(nsym, dtype=np.int64)
    for i in range(nsym):
        S[i] = gf_poly_eval(cw, gf_pow(_GEN, (fcr + i) % 255))
    return S


def _berlekamp_massey(S: np.ndarray, nsym: int) -> np.ndarray:
    """Error locator Lambda(x) from the syndrome sequence (little-endian)."""
    synd = [int(s) & 0xFF for s in S]
    C = np.zeros(nsym + 1, dtype=np.int64)
    B = np.zeros(nsym + 1, dtype=np.int64)
    C[0] = 1
    B[0] = 1
    L = 0
    m = 1
    b = 1
    for n in range(nsym):
        d = synd[n]
        for i in range(1, L + 1):
            if C[i] != 0 and (n - i) >= 0:
                d ^= gf_mul(int(C[i]), synd[n - i])
        if d == 0:
            m += 1
        elif 2 * L <= n:
            T = C.copy()
            coef = gf_div(d, b)
            for j in range(0, nsym + 1 - m):
                if B[j] != 0:
                    C[j + m] ^= gf_mul(coef, int(B[j]))
            L = n + 1 - L
            B = T
            b = int(d)
            m = 1
        else:
            coef = gf_div(d, b)
            for j in range(0, nsym + 1 - m):
                if B[j] != 0:
                    C[j + m] ^= gf_mul(coef, int(B[j]))
            m += 1
    return _trim(C[:L + 1])


def _chien_search(Lambda, nsym, n, fcr) -> list[int]:
    """Find error positions: Lambda(alpha^j) = 0 => error at coeff power
    ``p = -j (mod 255)`` (little-endian index)."""
    positions = []
    for j in range(0, 255):
        if gf_poly_eval(Lambda, gf_pow(_GEN, j)) == 0:
            p = (255 - j) % 255
            if 0 <= p < n:
                positions.append(p)
    return sorted(positions)


def _gf_gauss(A, b):
    """Solve A x = b over GF(256); A square. Returns solution vector or None
    if singular."""
    n = len(b)
    A = [[int(v) & 0xFF for v in row] for row in A]
    col = 0
    for r in range(n):
        # pivot
        piv = None
        for i in range(r, n):
            if A[i][col] != 0:
                piv = i
                break
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        b[r], b[piv] = b[piv], b[r]
        inv = gf_inv(A[r][col])
        for c in range(col, n):
            A[r][c] = gf_mul(A[r][c], inv)
        b[r] = gf_mul(b[r], inv)
        for i in range(n):
            if i != r and A[i][col] != 0:
                f = A[i][col]
                for c in range(col, n):
                    A[i][c] ^= gf_mul(f, A[r][c])
                b[i] ^= gf_mul(f, b[r])
        col += 1
        if col == n:
            break
    return np.array(b, dtype=np.int64)


def _magnitudes(positions, S, nsym, fcr) -> np.ndarray:
    """Error magnitudes by solving the e x e linear syndrome system
    ``sum_m mag_m * alpha^((fcr+i)*pos_m) = S[i]`` for i in 0..e-1."""
    e = len(positions)
    if e == 0:
        return np.zeros(0, dtype=np.int64)
    A = np.zeros((e, e), dtype=np.int64)
    b = np.array([int(s) & 0xFF for s in S[:e]], dtype=np.int64)
    for i in range(e):
        for m, p in enumerate(positions):
            A[i, m] = gf_pow(_GEN, ((fcr + i) * p) % 255)
    sol = _gf_gauss(A.tolist(), b)
    if sol is None or len(sol) != e:
        return np.zeros(e, dtype=np.int64)
    return sol


def rs_decode(received, params: RSParams):
    """Decode / correct ``params.t`` symbol errors (or fewer).

    Args:
        received: length-`params.n` GF(256) codeword (little-endian coeffs).
        params: RS profile.

    Returns:
        ``(decoded_msg, status)`` with ``decoded_msg`` the corrected length-`k`
        message and ``status``:
            >= 0  number of symbol errors corrected (0 = no errors)
            -1    decode failure (too many errors / uncorrectable)
    """
    r = np.asarray(received, dtype=np.int64).ravel()
    if r.size < params.n:
        r = np.concatenate([np.zeros(params.n - r.size, dtype=np.int64), r])
    elif r.size > params.n:
        raise ValueError(f"Received length {r.size} exceeds RS n={params.n}")

    n, k, t, fcr = params.n, params.k, params.t, params.fcr
    S = _syndromes(r[:n], params)
    if not np.any(S):
        return r[params.nsym:].copy(), 0

    Lambda = _berlekamp_massey(S, params.nsym)
    degree = len(Lambda) - 1
    while degree > 0 and Lambda[degree] == 0:
        degree -= 1
    if degree == 0 or degree > t:
        return r[params.nsym:].copy(), -1

    positions = _chien_search(Lambda, params.nsym, n, fcr)
    if len(positions) != degree or positions[-1] >= n:
        return r[params.nsym:].copy(), -1

    # Error magnitudes solved directly from the syndrome linear system.
    mags = _magnitudes(positions, S, params.nsym, fcr)
    if len(mags) != len(positions):
        return r[params.nsym:].copy(), -1

    decoded = r.copy()
    for pos, mag in zip(positions, mags):
        if mag:
            decoded[pos] = gf_add(int(decoded[pos]), int(mag))
    return decoded[params.nsym:].copy(), int(len(positions))


# ── Convenience API ─────────────────────────────────────────────────────────

def encode_rs(msg: np.ndarray, params: RSParams = RS255_223) -> np.ndarray:
    """Alias for :func:`rs_encode`."""
    return rs_encode(msg, params)


def decode_rs(codeword, params: RSParams = RS255_223):
    """Alias for :func:`rs_decode`."""
    return rs_decode(codeword, params)