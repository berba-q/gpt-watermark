"""Reduced version of the Gumbel-max watermark"""

import hashlib
import hmac
import math

# Demo secret key shared by the generator and detector.
# In practice, this should be a cryptographically secure random key, and the detector should not know it.
KEY = b"microgpt-demo-key"
CONTEXT = 5  # recent tokens used by the PRF


def uniform(context, token, key=KEY):
    """secret key + context + candidate token -> deterministic U(0, 1)"""
    msg = ','.join(map(str, context[-CONTEXT:])) + f'|{token}'
    digest = hmac.new(key, msg.encode(), hashlib.sha256).digest()
    integer = int.from_bytes(digest[:8], 'big')
    return (integer + 0.5) / 2**64    # never exactly 0 or 1


def sample(probabilities, context, key=KEY):
    """Keyed Gumbel-max sampling: argmax log(U_i) / p_i."""
    scores = []
    for token, p in enumerate(probabilities):
        u = uniform(context, token, key)
        scores.append(math.log(u) / p if p > 0 else -math.inf)
    return max(range(len(scores)), key=scores.__getitem__)


def microscope(probabilities, context, labels=None, key=KEY, top=8):
    """Print one token decision so the watermark is visible."""
    rows = []
    for token, p in enumerate(probabilities):
        u = uniform(context, token, key)
        score = math.log(u) / p if p > 0 else -math.inf
        rows.append((score, token, p, u))
    winner = max(rows)[1]
    rows = sorted(rows, key=lambda x: x[2], reverse=True)[:top]

    print("token       model p      keyed U     log(U)/p")
    print("------------------------------------------------")
    for score, token, p, u in rows:
        label = repr(labels[token]) if labels and token < len(labels) else str(token)
        mark = "  <-- selected" if token == winner else ""
        print(f"{label:8s}   {p:8.4f}   {u:10.7f}   {score:9.4f}{mark}")
    return winner
