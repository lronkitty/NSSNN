conda activate torch_37
python hsi_eval.py -a nssnn -ofn nssnn -s output \
--gpu-ids 2  -ofd results/nssnn/icvl/iid/50/ \
-r -rp checkpoints/icvl/iid.pth \
-tr datasets/test/ICVL/iid/50/ -gr datasets/test/ICVL/gt/

python hsi_eval.py -a nssnn -ofn nssnn -s output \
--gpu-ids 2  -ofd results/nssnn/icvl/iid/70/ \
-r -rp checkpoints/icvl/iid.pth \
-tr datasets/test/ICVL/iid/70/ -gr datasets/test/ICVL/gt/

python hsi_eval.py -a nssnn -ofn nssnn -s output \
--gpu-ids 2  -ofd results/nssnn/icvl/iid/90/ \
-r -rp checkpoints/icvl/iid.pth \
-tr datasets/test/ICVL/iid/90/ -gr datasets/test/ICVL/gt/
