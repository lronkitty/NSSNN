conda activate torch_37
python hsi_eval.py -a nssnn -ofn nssnn -s output \
--gpu-ids 2  -ofd results/nssnn/icvl/niid/15/ \
-r -rp checkpoints/icvl/niid.pth \
-tr datasets/test/ICVL/niid/15/ -gr datasets/test/ICVL/gt/

python hsi_eval.py -a nssnn -ofn nssnn -s output \
--gpu-ids 2  -ofd results/nssnn/icvl/niid/55/ \
-r -rp checkpoints/icvl/niid.pth \
-tr datasets/test/ICVL/niid/55/ -gr datasets/test/ICVL/gt/

python hsi_eval.py -a nssnn -ofn nssnn -s output \
--gpu-ids 2  -ofd results/nssnn/icvl/niid/95/ \
-r -rp checkpoints/icvl/niid.pth \
-tr datasets/test/ICVL/niid/95/ -gr datasets/test/ICVL/gt/
