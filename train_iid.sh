conda activate torch_37
python hsi_denoising_gauss_iid.py --batchSize 16 -a nssnn -p 2mats_iid \
--dataroot ./datasets/ICVL64_31_2mats.db --gpu-ids 3 \
-tr datasets/test/ICVL/iid/ -gr datasets/test/ICVL/gt/ \
--lr 1e-3 

python hsi_denoising_gauss_iid.py --batchSize 16 -a nssnn -p iid \
--dataroot ./datasets/ICVL64_31.db --gpu-ids 3 \
-tr datasets/test/ICVL/iid/ -gr datasets/test/ICVL/gt/ \
--lr 1e-3 \
-r -rp checkpoints/nssnn/2mats_iid/model_latest.pth \
--resetepoch 15