## ToDo

- Consider the transformer weight initialization of LVSM with std of $\frac{(0.02)}{(2(i+1))^{\frac{1}{2}}} = \frac{1}{50\sqrt{2(i+1)}}$ where $i$ is the index of the transformer layer
- Add suport for sampling more than 2 context views per batch.
- For both object and scene-level experiments,theviewselectiondetailsandcamera posenormalizationmethodsfollowGS-LRM.
- They train batch_size 64, 2 context, 6 target views -> We can only do 3 batches, 2 context, 4 target. Accumulate: 64/3

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

## Discussion
- Discuss presentation with Lukas