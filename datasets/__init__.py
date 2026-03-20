
from .toys import ToysDataset
from .home import HomeDataset 
from .home_category import HomeCategoryDataset
from .toys_category import ToysCategoryDataset

DATASETS = {
    ToysDataset.code(): ToysDataset,
    HomeDataset.code(): HomeDataset,
    HomeCategoryDataset.code(): HomeCategoryDataset,
    ToysCategoryDataset.code(): ToysCategoryDataset,
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
