python  tools/run_net.py --cfg configs/HOIVIT/EK_MF_ant.yaml TRAIN.ENABLE False TEST.ENABLE True MF.DEPTH 1 OUTPUT_DIR /data/usrdata/roy/checkpoints/mf_ek_ant_1l >> logs/1l_MF
python  tools/run_net.py --cfg configs/HOIVIT/EK_MF_ant.yaml TRAIN.ENABLE True TEST.ENABLE True MF.DEPTH 2 OUTPUT_DIR /data/usrdata/roy/checkpoints/mf_ek_ant_2l > logs/2l_MF
python  tools/run_net.py --cfg configs/HOIVIT/EK_MF_ant.yaml TRAIN.ENABLE True TEST.ENABLE True MF.DEPTH 4 OUTPUT_DIR /data/usrdata/roy/checkpoints/mf_ek_ant_4l > logs/4l_MF
python tools/run_net.py --cfg configs/HOIVIT/EK_HOIVIT_MF_doubleatt_ant.yaml TRAIN.ENABLE True TEST.ENABLE True MF.DEPTH 1 OUTPUT_DIR  checkpoints/hoivit_ek_wmotion_doubleatt_ant_1l > logs/InAViT_1l
python tools/run_net.py --cfg configs/HOIVIT/EK_HOIVIT_MF_doubleatt_ant.yaml TRAIN.ENABLE True TEST.ENABLE True MF.DEPTH 2 OUTPUT_DIR  checkpoints/hoivit_ek_wmotion_doubleatt_ant_2l > logs/InAViT_2l
python tools/run_net.py --cfg configs/HOIVIT/EK_HOIVIT_MF_doubleatt_ant.yaml TRAIN.ENABLE True TEST.ENABLE True MF.DEPTH 4 OUTPUT_DIR  checkpoints/hoivit_ek_wmotion_doubleatt_ant_4l > logs/InAViT_4l
