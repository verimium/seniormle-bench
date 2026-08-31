# Intended Improvement: Broaden Negative Exposure Within Each Training Block

## Production Baseline

The agent starts from the former clean reference implementation, now presented
as the deployed production model. It already has the correct chronological
target window, full-history false-negative filtering, explicit preference state,
organic target weighting, a GRU user tower, and an independently indexable item
tower. There is no planted temporal or data-leakage bug.

The production objective uses 24 uniform negatives and eight
propensity-corrected unique in-batch negatives per query. The eight in-batch
items are selected from positive targets elsewhere in the logical training
block, but most of the block's observed positives are not compared with a given
query. This leaves useful discrimination signal unused after the item tower has
already computed those positive representations.

## Reference Improvement

### Paper Reference

Jinpeng Wang, Jieming Zhu, and Xiuqiang He. "Cross-Batch Negative Sampling for
Training Two-Tower Recommenders." *Proceedings of the 44th International ACM
SIGIR Conference on Research and Development in Information Retrieval*, 2021,
pp. 1632-1636. DOI: [10.1145/3404835.3463032](https://doi.org/10.1145/3404835.3463032).
Open preprint: [arXiv:2110.15154](https://arxiv.org/abs/2110.15154).

### Lesson Applied Here

The paper's core lesson is that negative-sampling coverage in a two-tower model
need not be limited by the physical mini-batch size. Once item representations
have already been encoded, reusing representations from nearby batches can
provide many more query-item comparisons at much lower cost than increasing the
batch and re-encoding all features. The paper justifies a recent-batch cache by
observing that item embeddings become sufficiently stable after warm-up.

Our reference applies the broader compute-reuse lesson, not the paper's exact
algorithm. On this small CPU workload, all positive targets in a logical
training block are available together, so the implementation shares their
attached embeddings within that block. It avoids a stale detached cache while
retaining false-negative filtering and correction for the non-uniform item
sampling distribution.

The improved reference keeps the production architecture, serving score, target
weights, uniform-negative term, and relative 24-to-8 loss balance. During each
training microbatch it also:

1. Collects up to 512 unique current positive target items.
2. Reuses their attached item vectors as a shared negative candidate pool.
3. Scores every query against that pool with one matrix multiplication.
4. Masks the query's target, all known positives in the fitting history, and
   active likes before computing the shared negative loss.
5. Applies the existing item-frequency correction to reduce the popularity bias
   of positives reused as negatives.

The pool's embeddings remain attached to autograd, so candidate items learn from
both their positive role and valid negative comparisons. A separate seeded
random stream downsamples pools larger than 512 without perturbing the existing
uniform-negative sampler.

This is not a direct implementation of the paper's cross-batch FIFO cache. A
detached FIFO pool with the existing BCE adaptation scored
`0.007837` public NDCG@10, and a corrected sampled-softmax version scored
`0.006312`; both were rejected. The accepted version avoids stale embeddings and
does not change the model's BCE objective or serving score.

## What The Task Examines

The task asks whether an agent can reason about:

- negative-sampling coverage in a retrieval model;
- the difference between sampled negatives and known user positives;
- sampling-distribution correction for popularity-biased in-batch items;
- reusing item-tower computation without violating the indexed serving path;
- attached versus stale or detached candidate embeddings; and
- validation-driven model improvement under a fixed CPU budget.

The paper or method name is not supplied to the agent. The production code and
problem context expose the modeling bottleneck, but agents may implement any
sound two-tower improvement that clears the hidden metric. There is no private
source-pattern or paper-reproduction gate.

## Recorded Metrics And Threshold

These author records come from the Harbor Linux CPU containers using seed 29
and the same public-data-only protocol for both implementations. Native PyTorch
builds can differ slightly in their exact rankings. Public validation is the
development signal; it is not a separate acceptance gate.

| Implementation | Public validation NDCG@10 | Hidden test NDCG@10 |
| --- | ---: | ---: |
| Production two-tower | 0.021280 | 0.016196 |
| Shared-negative reference | 0.022693 | 0.016939 |
| Nominal reference threshold | n/a | 0.016939 |
| Effective gate after 1% tolerance | n/a | 0.01676961 |

The production model misses the effective hidden gate by `0.00057361`, or about
3.54% relative to its hidden score. The improved reference clears it by
`0.00016939`. Since NDCG@10 is reported to six decimals, `0.016770` is the
smallest reported value that passes.

## Automated Acceptance

The verifier:

1. Validates ranking schema, user coverage, ranks, item uniqueness, catalog
   membership, and cutoff-correct candidate eligibility.
2. Scores hidden-test NDCG@10 and requires at least `0.01676961`.

The measured NDCG remains the dense reward. The verifier does not inspect source
text, require a named algorithm, or evaluate the experiment README. Hidden
labels and evaluator-owned artifacts remain isolated from model-building code.

## Private Reference Sources

The improved implementation is owned by this case at
`reference_solution/reference_solver.py`; the parent task retains its original
reference solver. During task generation, the case overlay materializes the
former reference as `production_solver.py` and installs it publicly as
`current_model/train_current_model.py`. The overlay is hash-guarded, and
`test_shared_negative_pool.py` checks that production and reference retain the
same chronological partition while differing in shared-negative capability.
The established Harbor results and corresponding source hashes for both models
are recorded in `model_metrics.json`, so routine package builds do not retrain
either calibration model.
