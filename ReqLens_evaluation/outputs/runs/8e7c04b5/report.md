# Evaluation Report — Run `8e7c04b5`

**Created:** 2026-05-27T23:15:40.191652+00:00  
**Systems:** reqinone_v1, reqlens_v2  
**Tracks:** 1, 2

## Track 1 — Trustworthiness (UAR / HRR / GRR)

| System | Units | Mean UAR ↓ | Mean HRR ↑ | Mean GRR ↓ |
|--------|-------|-----------|-----------|-----------|
| `reqinone_v1` | 13 | 0.0000 | 1.0000 | 0.1980 |
| `reqlens_v2` | 13 | 0.0000 | 1.0000 | 0.0424 |

## Track 2 — Defect Detection (DLR)

| System | Units | Mean DLR ↓ |
|--------|-------|-----------|
| `reqinone_v1` | 13 | 0.3462 |
| `reqlens_v2` | 13 | 0.2077 |

---
> UAR (Unsupported Acceptance Rate): lower is better — fraction of hallucinations the system incorrectly accepted.
> HRR (Hallucination Rejection Rate): higher is better — fraction of hallucinations the system correctly rejected.
> GRR (Gold Rejection Rate): lower is better — fraction of legitimate gold requirements the system incorrectly rejected.
> DLR (Defect Leakage Rate): lower is better — fraction of seeded defects that leaked into the system's extraction output.