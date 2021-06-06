#!/usr/bin/env bash
cd src && \
python main.py --task show --config configs/atari_riverraid.yaml resume True resume_ckpt '../pretrained/atari_riverraid.pth' device 'cuda:0' show.indices "[0, 1, 2, 3, 4]"

