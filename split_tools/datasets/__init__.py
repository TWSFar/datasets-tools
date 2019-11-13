from datasets import visdrone


def get_dataset(db_name, db_root, split):
    if db_name in ['VisDrone', 'visdrone', 'VisDrone']:
        return visdrone.VisDrone(db_root, split)
    else:
        raise NotImplementedError
