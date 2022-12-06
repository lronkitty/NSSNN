conda activate torch_37
python hsi_denoising_gauss_niid.py --batchSize 16 -a nssnn -p 2mats_niid \
--dataroot ./datasets/ICVL64_31_2mats.db --gpu-ids 0 \
-tr datasets/test/ICVL/niid/ -gr datasets/test/ICVL/gt/ \
--lr 1e-3 

python hsi_denoising_gauss_niid.py --batchSize 16 -a nssnn -p niid \
--dataroot ./datasets/ICVL64_31.db --gpu-ids 4 \
-tr datasets/test/ICVL/niid/ -gr datasets/test/ICVL/gt/ \
--lr 1e-3 \
-r -rp checkpoints/nssnn/2mats_niid/model_latest.pth \
--resetepoch 15