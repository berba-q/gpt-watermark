# GPT Watermark

Experiment gpt watermarking based on Karpathy's microgpt

Built on Andrej Karpathy's minimal `microgpt` to show exactly how a secret-keyed sampler can create a statistically detectable watermark without changing the model's underlying token probabilities.

- Pure python
- No dependency libraries
- Few lines of code

## The idea

An LLM generates text by sampling the next token from a probability distribution.

Normal generation:

GPT → probabilities → random sampling → next token

Watermarked generation:

GPT → same probabilities → keyed Gumbel sampling → next token

The watermark is not a visible marker in the text. It emerges statistically across many token choices and can be detected using the same secret key.

## Try it

```bash
git clone https://github.com/berba-q/gpt-watermark
cd gpt-watermark
python3 microgpt_watermark.py
```

No dependencies beyond the Python 3 standard library.

## Results

Same tiny GPT, same probabilities, two samplers — the text looks equally natural either way:

```text
yuh | normal: yuha           | watermarked: yuhan
xav | normal: xavinn         | watermarked: xavia
jua | normal: juan           | watermarked: juale
```

But the detector, holding the secret key, tells them apart with overwhelming confidence:

```text
normal       n=308  mean=0.962  p=7.459e-01
watermarked  n=322  mean=2.105  p=6.844e-53
wrong key    n=322  mean=0.984  p=6.094e-01
```

Watermarked text scores a p-value of ~6.8×10⁻⁵³; ordinary text and text checked with the wrong key both land around p≈0.6-0.75 indistinguishable from chance.

Want the full explanation? [Read: Watermarking a tiny GPT: 91 lines, NO Frameworks](https://berba-q.github.io/blog/llm-watermark-microgpt.html).

## Acknowledgments

This project builds on Andrej Karpathy's excellent [`microgpt`](https://karpathy.github.io/2026/02/12/microgpt/), which provides the minimal GPT implementation used here. The watermarking experiment is inspired by [Scott Aaronson's work](https://scottaaronson.blog/?p=9333) on keyed Gumbel sampling for LLM output watermarking.
