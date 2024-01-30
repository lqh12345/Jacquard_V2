def get_dataset(dataset_name):
    if dataset_name == 'cornell':
        from .cornell_data import CornellDataset
        return CornellDataset
    elif dataset_name == 'jacquard':
        from .jacquard_data import JacquardDataset
        return JacquardDataset
    elif dataset_name == 'multi_targets':
        from .multi_targets import Multi_Targets_Dataset
        return Multi_Targets_Dataset
    else:
        raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))