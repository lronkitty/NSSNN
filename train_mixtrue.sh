conda activate torch_37
python hsi_denoising_complex.py --batchSize 16 -a nssnn -p mxiture \
--dataroot ./datasets/ICVL64_31.db --gpu-ids 2 \
-tr datasets/test/ICVL/mixture/95_mixture/ -gr datasets/test/ICVL/gt/ \
--lr 1e-4 \
-r -rp checkpoints/nssnn/niid/model_latest.pth \
--resetepoch 50

