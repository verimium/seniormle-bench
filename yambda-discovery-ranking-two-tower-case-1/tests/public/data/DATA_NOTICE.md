# Data notice

The data files in this directory are derived from the **Yambda-5B** dataset,
Copyright Yandex, licensed under the Apache License, Version 2.0.

- Source: <https://huggingface.co/datasets/yandex/yambda>
- Paper: <https://arxiv.org/abs/2505.22238>

**These files have been modified.** They were subsampled to a small fraction of
the original interaction log, re-split into train / validation / test partitions
on a different time boundary, re-keyed, filtered to a restricted candidate
catalog, and reformatted as benchmark task fixtures. Eligibility and label
artifacts (`*_eligible_candidates.npz`, `*_preference_state.parquet`, and the
private truth files used by the verifier) were computed here and do not appear
in the upstream dataset.

They are not the original Yambda-5B dataset and must not be used as a substitute
for it. Obtain the original from the source link above.

The upstream dataset card states that Yambda-5B is published for scientific and
research purposes.

## Cite as

> Ploshkin, A., Tytskiy, V., Pismenny, A., Baikalov, V., Taychinov, E.,
> Permiakov, A., Burlakov, D., Krofto, E., Savushkin, N.
> *Yambda-5B — A Large-Scale Multi-modal Dataset for Ranking And Retrieval.*
> arXiv:2505.22238, 2025. <https://arxiv.org/abs/2505.22238>
