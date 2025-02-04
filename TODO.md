# ToDo


## Fix
- Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 1. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
- Your `IterableDataset` has `__len__` defined. In combination with multi-process data loading (when num_workers > 1), `__len__` could be inaccurate if each worker is not configured independently to avoid having duplicate data.
- Analyze the error of flex-attention not running when using the re10k_vae mode.

## Verify
- Verify/Tune the gradient clipping limits (currently only based on norm not on absolute value)
- Verify attention masking (src->src, tgt->tgt, tgt->src)

## Code Cleanup
- Cleanup leftover MVSplat configurations and non-compatible functions

## Optional Enhancements
- Continue the same wandb run in case a checkpoint is loaded that started from a run before
- Consider the transformer weight initialization of LVSM with std of $\frac{(0.02)}{(2(i+1))^{\frac{1}{2}}} = \frac{1}{50\sqrt{2(i+1)}}$ where $i$ is the index of the transformer layer