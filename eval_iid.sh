conda activate torch_37
python hsi_eval.py -a nssnn -ofn nssnn -s output \
--gpu-ids 2  -ofd results/nssnn/icvl/iid/50/ \
-r -rp checkpoints/icvl/iid.pth \
-tr testsets/ICVL/iid/50/ -gr testsets/ICVL/gt/

python hsi_eval.py -a nssnn -ofn nssnn -s output \
--gpu-ids 2  -ofd results/nssnn/icvl/iid/70/ \
-r -rp checkpoints/icvl/iid.pth \
-tr testsets/ICVL/iid/70/ -gr testsets/ICVL/gt/

python hsi_eval.py -a nssnn -ofn nssnn -s output \
--gpu-ids 2  -ofd results/nssnn/icvl/iid/90/ \
-r -rp checkpoints/icvl/iid.pth \
-tr testsets/ICVL/iid/90/ -gr testsets/ICVL/gt/