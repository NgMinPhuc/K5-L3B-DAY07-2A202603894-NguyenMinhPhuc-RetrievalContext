# Day 7 dev selection

**Selected config:** `hybrid_rrf_more_context` (hybrid RRF, candidate pool 20 per retriever, top-k 5, context budget 1,400 characters).

| Config | Evidence Hit@3 | Context evidence | Answered (out of 160 answerable) | Answer grade 0/1/2 | Estimated cost |
|---|---:|---:|---:|---|---:|
| TF-IDF baseline | 103/160 | 45/160 | 19 | Not manually reviewed | $0.3510 |
| Dense baseline | 122/160 | 42/160 | 17 | Not manually reviewed | $0.3623 |
| Hybrid RRF baseline | 139/160 | 51/160 | 21 | 141 / 1 / 18 | $0.3476 |
| Hybrid RRF, candidate pool 40 | 137/160 | 53/160 | 21 | Not manually reviewed | $0.3427 |
| **Hybrid RRF, top-k 5 + context 1,400** | **135/160** | **101/160** | **27** | **136 / 0 / 24** | **$0.3116** |

## Why this config was selected

Hybrid RRF with the larger context loses 4 Evidence Hit@3 cases versus the hybrid baseline, while final-context evidence rises from 51/160 to 101/160 and the number of answered answerable questions rises from 21 to 27. Its manually reviewed answer-grade mean is 0.30/2 versus 0.23/2 for the hybrid baseline. Estimated API cost is also lower in this run. This favors getting evidence into the final model context and producing more grounded answers; retrieval-only Hit@3 remains slightly higher for the baseline.

The selected config correctly abstained on both dev questions without evidence (2/2). Of its 27 answered cases, manual review found 24 correct and supported by citation, and 3 incorrect/unsupported: `q048`, `q104`, and `q105`. All 27 citation IDs appeared in the final context, but ID presence alone did not establish support. Answerable abstentions remain frequent (133/160), so the controller prompt and search behavior need further improvement.

The hybrid baseline also exposed two errors: `q065` retrieved the older 2012 rule while the benchmark expects the 2019 rule, and `q199` cited a strike procedure instead of answering the requested arbitration deadline. `q035` was partially complete because it omitted the explicit written-consent condition.

The 40-question Day06 test split was not read by this experiment. Selection is provisional on dev and should be frozen before running Day07 test. Estimated cost is based on token usage and the configured model prices, not a billing statement.
