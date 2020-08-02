from torch.utils.data import DataLoader

from datasets import find_dataset_using_name
from options.train_options import TrainOptions

if __name__ == "__main__":

    # test me with PYTHONPATH='.' test/test_dataset.py --dataset [name], from the project root

    opt = TrainOptions().parse()
    dataset = find_dataset_using_name(opt.dataset)(opt)
    print(f"{dataset = }")

    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.workers,
        shuffle=not opt.no_shuffle,
    )

    print(
        f"Size of the dataset: {len(dataset):05d}, "
        f"dataloader: {len(dataloader):04d}"
    )
    first_item = dataset[0]
    first_batch = next(iter(dataloader))

    from IPython import embed

    embed()
