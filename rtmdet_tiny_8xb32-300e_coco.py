_base_ = './configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

model = dict(
    backbone=dict(
        deepen_factor=0.167,
        widen_factor=0.375,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(in_channels=96, feat_channels=96, exp_on_reg=False))

# Modify dataset related settings
data_root = './customDataset/COCO_format/'
metainfo = {
    'classes': ('balloon', ),
    'palette': [
        (220, 20, 60),
    ]
}
classes = (
        'Motor Vehicle',
        'Non-motorized Vehicle',
        'Pedestrian',
        'Traffic Light-Red Light',
        'Traffic Light-Yellow Light',
        'Traffic Light-Green Light',
        'Traffic Light-Off'
        )
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='train/instances.json',
        data_prefix=dict(img='train/images/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/instances.json',
        data_prefix=dict(img='val/images/')))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'val/instances.json')
test_evaluator = val_evaluator
