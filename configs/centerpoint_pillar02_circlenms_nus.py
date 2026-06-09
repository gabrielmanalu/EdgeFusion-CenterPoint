# Project-level config wrapper.
# Points to the mmdet3d config used for the open-mmlab CenterPoint baseline.
#
# Upstream config:
#   /workspace/mmdetection3d/configs/centerpoint/
#       centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py

_base_ = [
    "/workspace/mmdetection3d/configs/centerpoint/"
    "centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"
]

# Override data root to local extraction path
data_root = "/data/nuscenes/"
