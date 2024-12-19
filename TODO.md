## ToDo

- Consider the transformer weight initialization of LVSM with std of $\frac{(0.02)}{(2(i+1))^{\frac{1}{2}}} = \frac{1}{50\sqrt{2(i+1)}}$ where $i$ is the index of the transformer layer
- Add suport for sampling more than 2 context views per batch.
- For both object and scene-level experiments,theviewselectiondetailsandcamera posenormalizationmethodsfollowGS-LRM.
- They train batch_size 64, 2 context, 6 target views -> We can only do 3 batches, 2 context, 4 target. Accumulate: 64/3

## Fix
- Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 1. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
- /home/team15/lvsm_02/src/visualization/camera_trajectory/interpolation.py:82: UserWarning:
  Using torch.cross without specifying the dim arg is deprecated.
  Please either pass the dim explicitly or simply use torch.linalg.cross.
  The default value of dim will change to agree with that of linalg.cross in a future release. (Triggered internally at /pytorch/aten/src/ATen/native/Cross.cpp:62.)
- Your `IterableDataset` has `__len__` defined. In combination with multi-process data loading (when num_workers > 1), `__len__` could be inaccurate if each worker is not configured independently to avoid having duplicate data.

## Training Ideas
- Disable Augmentation
- Review larger gradient clipping
- Review structure for motion camera paths

## Verify
- Verify Plucker Ray implementation
- Verify correct checkpointing/loading e.g. is the state fully restored?
- Verify/Tune the gradient clipping limits (currently only based on norm not on absolute value)
- Verify attention masking (src->src, tgt->tgt, tgt->src)

## Code Cleanup
- Remove commented ray shims from dataloader if gpu works better
- Cleanup leftover MVSplat configurations and non-compatible functions

## Architectural Questions
- Does LVSM also exclude biases on the linear tokenize/untokenize projections?

## Optional Enhancements
- Continue the same wandb run in case a checkpoint is loaded that started from a run before

## Tests
- Scale down transformer to learn faster
- Train such that context and targets are the same, to learn the identity -> Done! Seems okay.
- Turn off mixed precision to rule out gradient issues
- Check scale again on multiple scenes
- Check magnitude of Plucker rays -> Done! Seems okay.
- Plot attention maps
- Run tests on ssd with smaller image size
- Save preprocessed (cropped / resized) images on ssd