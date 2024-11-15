## ToDo

- FlashAttention2 although masking required -> Use xformers or FlexAttention or PyTorch kernels
- Add support for QK-Norm in our Model
- Consider the transformer weight initialization of LVSM with std of $\frac{(0.02)}{(2(i+1))^{\frac{1}{2}}} = \frac{1}{50\sqrt{2(i+1)}}$ where $i$ is the index of the transformer layer
- Add 3D visualisations of the cameras in space along with the image planes (PyTorch 3D / Open3D) - Check for correct parametrization (OpenCV vs. PyTorch 3D)
- Add suport for sampling more than 2 context views per batch.
- Add mse and total loss to validation wandb charts

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