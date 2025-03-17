
Code for running the scripts with `nohup`:

```{sh}
conda deactivate
conda activate uni2h_env
nohup nice -n 10 python3 uni2_h_extractor.py > nohup_extractor.out &
```