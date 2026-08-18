"""Statistical detection for the tiny Gumbel-max watermark."""

# LIBRARIES
import math
from watermark import KEY, CONTEXT, uniform


def token_score(token, context, key=KEY):
    """Function of the keyed uniform random number for a token, given its context."""
    # Under no watermark U is uniform, so -log(1-U) is Exponential(mean=1).
    return -math.log1p(-uniform(context, token, key))


def gamma_tail(score, n):
    """Function to compute P[Gamma(n, 1) >= score] for integer n, using the log-sum-exp trick to avoid underflow."""
    if n == 0:
        return 1.0
    terms = [0.0 if k == 0 else k * math.log(score) - math.lgamma(k + 1)
             for k in range(n)] if score > 0 else [0.0] + [-math.inf] * (n - 1)
    m = max(terms)
    log_p = -score + m + math.log(sum(math.exp(x - m) for x in terms))
    return math.exp(log_p) if log_p > -745 else 0.0


def detect(samples, key=KEY):
    """
    Function to compute the mean score and p-value for a set of token sequences, given the secret key.
    Return (n, mean score, p-value) across token sequences.

    samples is an iterable of (tokens, first_generated_token_index).
    Repeated contexts are counted once: the PRF would otherwise reuse the
    same pseudorandom numbers and create fake statistical confidence.
    """
    score = 0.0
    n = 0
    seen = set()
    for tokens, start in samples:
        for i in range(start, len(tokens)):
            context = tuple(tokens[max(0, i - CONTEXT):i])
            if context in seen:
                continue
            seen.add(context)
            score += token_score(tokens[i], context, key)
            n += 1
    return n, score / n if n else 0.0, gamma_tail(score, n)
