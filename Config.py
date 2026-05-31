class Config:
    # --- data ---
    data_csv:    str = "/home/simon/Documents/Zwischen Wörtern und Pixeln/annotations-caer_cleaned_4cluster.csv"
    image_col:   str = "image_path"
    label_col:   str = "emotion_recode2"   # 0 = Optimistic; 1 = Pessimistic; 2 = Hostile; 3 = Neutral
    num_classes: int = 4
    img_size:    int = 224
    val_ratio:  float = 0.2
    random_seed: int = 42
    imagenet_mean: tuple = (0.485, 0.456, 0.406) #standard imagenet
    imagenet_std: tuple = (0.229, 0.224, 0.225) #standard imagenet
    EMOTION_NAMES: list = ["Optimistic", "Pessimistic", "Hostile", "Neutral"]

    # --- model ---
    model_name:     str   = "swin_base_patch4_window7_224"
    pretrained:     bool  = True
    drop_rate:     float  = 0.3    #dropout before the classification head
    drop_path_rate: float = 0.1    #stochastic depth, enabled at Phase 2b only
    batch_size: int = 64

    # --- training phases ---
    #Phase 1  : backbone frozen, head only, no MixUp
    epochs_frozen:   int   = 3
    lr_head:        float  = 1e-4

    #Phase 2a : backbone still frozen, MixUp enabled, head consolidates
    epochs_head:     int   = 2
    lr_head_warmup: float  = 5e-5   #slightly lower than Phase 1 for stability

    #Phase 2b : full fine-tune, conservative backbone LR
    epochs_finetune: int   = 30     #early stopping might cut this
    lr_backbone:    float  = 5e-6
    lr_head_ft:     float  = 2e-5

    weight_decay:   float  = 1e-4
    grad_clip:      float  = 1.0

    # --- MixUp ---
    #Disabled in Phase 1 and first epoch of Phase 2b.
    #Enabled in Phase 2a onward.
    mixup_alpha: float = 0.2

    # --- early stopping ---
    patience: int = 7

    # --- inference ---
    threshold: float = 0.6 #if confidence is below threshold, label "unknown" is cast

    # --- system ---
    num_workers: int = 8
    device:      str = "cuda"

cfg = Config()

