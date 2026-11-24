from src.utils.config import load_config
from src.utils.seed import set_seed
from src.dataset.loader import build_loaders


if __name__ == "__main__":
    # Config 로드 및 Seed 설정
    config = load_config("configs/ntu120_skeleton.yaml")
    set_seed(config["experiment"]["seed"])

    # DataLoader 생성
    loaders = build_loaders(config)

    train_loader, val_loader = loaders[:2]

    print("Train samples:", len(train_loader.dataset))
    print("Val samples:", len(val_loader.dataset))

    if len(loaders) == 3:
        print("Test samples:", len(loaders[2].dataset))

    # Dataset에서 Skeleton 샘플 하나 로드
    x, label = train_loader.dataset[0]

    print("Sample shape:", x.shape)
    print("Frame count:", x.shape[1])
    print("Action ID:", label.item() + 1)

    # DataLoader에서 배치 하나 로드
    batch_x, batch_labels = next(iter(train_loader))

    print("Batch shape:", batch_x.shape)
    print("Batch labels:", batch_labels.shape)
