conda activate torch_37
python hsi_eval.py -a nssnn -ofn nssnn -s output \
--gpu-ids 2  -ofd results/nssnn/icvl/niid/15/ \
-r -rp checkpoints/icvl/niid.pth \
-tr testsets/ICVL/niid/15/ -gr testsets/ICVL/gt/

python hsi_eval.py -a nssnn -ofn nssnn -s output \
--gpu-ids 2  -ofd results/nssnn/icvl/niid/55/ \
-r -rp checkpoints/icvl/niid.pth \
-tr testsets/ICVL/niid/55/ -gr testsets/ICVL/gt/

python hsi_eval.py -a nssnn -ofn nssnn -s output \
--gpu-ids 2  -ofd results/nssnn/icvl/niid/95/ \
-r -rp checkpoints/icvl/niid.pth \
-tr testsets/ICVL/niid/95/ -gr testsets/ICVL/gt/