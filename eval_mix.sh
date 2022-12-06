conda activate torch_37
python hsi_eval.py -a nssnn -ofn nssnn -s output \
--gpu-ids 2  -ofd results/nssnn/icvl/mixture/95_mixture/ \
-r -rp checkpoints/icvl/mixture.pth \
-tr testsets/ICVL/mixture/95_mixture/ -gr testsets/ICVL/gt/