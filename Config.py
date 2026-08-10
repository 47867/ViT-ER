class Config:

    #data
    data_csv_train: str = "/home/simon/Documents/Zwischen Wörtern und Pixeln/annotations-caer_cleaned_4cluster.csv"
    data_csv_adaptation: str = "/home/simon/Documents/Zwischen Wörtern und Pixeln/Sample_Dataset/Reichel/domain_adaptation_coded_weighted.csv"
    image_col: str = "image_path"
    label_col: str = "emotion_recode2"   # 0 = Optimistic; 1 = Pessimistic; 2 = Hostile; 3 = Neutral
    num_classes: int = 4
    img_size: int = 224
    val_ratio: float = 0.2
    random_seed: int = 42
    imagenet_mean: tuple = (0.485, 0.456, 0.406) #standard imagenet
    imagenet_std: tuple = (0.229, 0.224, 0.225) #standard imagenet
    EMOTION_NAMES: list = ["Optimistic", "Pessimistic", "Hostile", "Neutral"]
    label_map: dict = {
        "Optimistic": 0,
        "Pessimistic": 1,
        "Hostile": 2,
        "Neutral": 3}

    #model
    model_name: str = "swin_base_patch4_window7_224" #choose desired model. for domain adaptation also choose layers to unfreeze below
    #model_name: str = "deit_base_patch16_224"
    pretrained: bool = True
    drop_rate: float = 0.3    #dropout before the classifier head
    drop_path_rate: float = 0.1    #stochastic depth, enabled at Phase 2b only
    batch_size_main: int = 64

    #main training phases
    #main: phase 1  : backbone frozen, head only, no MixUp
    epochs_frozen: int   = 3
    lr_head: float  = 1e-4

    #main: phase 2a : backbone still frozen, MixUp enabled, head consolidates
    epochs_head: int   = 2
    lr_head_warmup: float  = 5e-5   #slightly lower than Phase 1 for stability

    #main: phase 2b : full fine-tune, conservative backbone LR
    epochs_finetune: int   = 30     #early stopping might cut this
    lr_backbone: float  = 5e-6
    lr_head_ft: float  = 2e-5

    weight_decay: float  = 1e-4
    grad_clip: float  = 1.0

    #MixUp
    #disabled in Main Phase 1 and first epoch of Main Phase 2b.
    #enabled in Main Phase 2a onward.
    mixup_alpha: float = 0.2

    #early stopping
    patience: int = 7

    #inference
    threshold: float = 0.4 #if confidence is below threshold, label "unknown" is cast

    #system
    num_workers: int = 8
    device: str = "cuda" #NVIDIA Blackwell GB203 accelerator

    #domain adaptation
    stage: int = 2 #change according to desired adaptation
    stage_config: dict = {
    1: {"description": "Head only, backbone fully frozen",
        "lr_head": 1e-4,
        "lr_backbone": 0.0,
        "epochs": 100,
        "patience": 25, #be patient
        "unfreeze_layers": [],
        "min_images": 150},

    2: {"description": "Head + last layers unfrozen",
        "lr_head": 5e-5,
        "lr_backbone": 5e-6, #keep small to avoid overfitting
        "epochs": 80,
        "patience": 15, #be patient

        #layers.3 are for SWIN-Base and blocks.x are for DeiT-Base. choose together with model
        "unfreeze_layers": ["layers.3"], #can add layers.2 for more params, but model might overfit
        #"unfreeze_layers": ["blocks.7", "blocks.8", "blocks.9", "blocks.10", "blocks.11"],
        "min_images": 300}}
    batch_size_adaptation: int = 16 #keep small
    num_workers_adaptation: int = 4
    weight_decay_adaptation: float = 1e-3
    label_smoothing_adaptation: float = 0.00 #set to zero for now
    mixup_alpha_adaptation: float = 0.125 #best performance, empirically tested


#now we can import Config as cfg
cfg = Config()

