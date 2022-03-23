from dataset import LAHeart
from torch.utils.data import DataLoader


if __name__ == '__main__':
    max_epoch = 1000
    batch_size = 2

    model = None
    train_dst = LAHeart(split='train', transform=None)
    # test_dst = LAHeart(split='test', transform=None)

    train_loader = DataLoader(
        train_dst, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True
    )
    
    optimizer = None
    for epoch in range(max_epoch):
        model.train()
        for batch in train_loader:
            pass
