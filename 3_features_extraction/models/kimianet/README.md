
Code for running the scripts with `nohup`:

```{sh}
conda deactivate
conda activate example_kimianet_11_8
nohup nice -n 10 python3 kimianet_extractor_with_patch_width.py > nohup_extractor_with_width.out &
```