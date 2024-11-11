- FlashAttention2 although masking required -> Use xformers or flexattention or pytorch kernels
- Check bias for token linear layers in lvsm model
- Set Bfloat16 as type for our model
- Exclude LayerNorms from optimizer weight decay
- Check for their weight init
- Check for gradient clipping
- Remove commented ray shims from dataloader if gpu works better
- Check if checkpointing and loading works
- Add QK-Norm

Questions:
- Plucker ray computation implemented correctly?
- Talk about LPIPS loss
- Talk about attention masking (src->src, tgt->tgt, tgt->src)
- Storage space on SL21 / SL08 compute nodes
- Upgrade cuda version from 12.2 to 12.4 or allow in parallel