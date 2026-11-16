# Advanced dev selection

Selected (frozen before test): **v2_wide** — hybrid RRF (dense text-embedding-3-small + TF-IDF), top-k 8, context 10,000 chars, max 6 searches, index v2 (annex section headers + continued-table labels), label boost, prompt v2.

| Config | Pass (AI-assisted) | By group | Gold pages cited | Chat cost | Mean latency |
|---|---:|---|---:|---:|---:|
| text_only | 12/23 | table 3/6, chart 1/6, figure 1/2, multihop 2/3, text 2/3, abstain 3/3 | 13/20 | $0.1031 | 4.2 s |
| multimodal | 17/23 | table 4/6, chart 5/6, figure 1/2, multihop 2/3, text 2/3, abstain 3/3 | 14/20 | $0.0919 | 3.7 s |
| v2_wide | 22/23 | table 6/6, chart 6/6, figure 2/2, multihop 2/3, text 3/3, abstain 3/3 | 18/20 | $0.118 | 3.5 s |
| v2_wide_pageimg | 20/23 | table 5/6, chart 6/6, figure 1/2, multihop 2/3, text 3/3, abstain 3/3 | 18/20 | $0.2643 | 4.7 s |

Not graded for pass/fail (retrieval metrics only, see summaries): multimodal_dense (15/20 gold pages cited), multimodal_wide (14/20).

Why: v2_wide has the highest dev pass rate; adding page images (v2_wide_pageimg) cost ~2.2x and introduced chart misreadings (A-dev-05) and self-contradiction (A-dev-01). Remaining dev failure B-dev-06: wrong assumption, page 51 not cited.
Test split not run or read before this selection.
