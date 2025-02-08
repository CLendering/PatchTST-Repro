# TODO List 


- [ ] Refactor `utils/callback/`
- [ ] Refator `pre-train/`
- [ ] Refactor `data/pred_dataset`
- [ ] Refactor `utils/learner.py`
- [ ] Refactor `utils/utils.py`
- [ ] Merge functions in `data` and `utils` to avoid redundancy

## Improvements

1. Cross-Channel Attention: Selectively model dependencies between time series variables.
2. Efficient Attention Mechanisms: Swap vanilla self-attention for Performer/Linformer to cut complexity.
4. Adaptive Normalization: Use running stats or AdaNorm to handle concept drift better.
5. Smarter Masking: Span-wise masking for harder self-supervised learning.
6. Real world Benchmarks: Test on climate, biomedical datasets etc. for real-world validation.

