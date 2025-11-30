from provider.nocs_dataset import TrainingDataset, FinetuneDataset, FinetuneOnREAL275Dataset
from provider.housecat6d_dataset import HouseCat6DTrainingDataset
import torch

def create_dataloaders(cfg):
    data_dir = cfg.dataset_dir
    data_loader = {}

    if cfg.dataset_name == "camera_real":
        syn_dataset = TrainingDataset(
            cfg.image_size, cfg.sample_num, data_dir, 'syn',
            num_img_per_epoch=cfg.num_mini_batch_per_epoch*cfg.syn_bs, threshold=cfg.outlier_th)
            
        syn_dataloader = torch.utils.data.DataLoader(syn_dataset,
            batch_size=cfg.syn_bs,
            num_workers=cfg.syn_num_workers,
            shuffle=cfg.shuffle,
            sampler=None,
            drop_last=cfg.drop_last,
            pin_memory=cfg.pin_memory)
            
        real_dataset = TrainingDataset(
            cfg.image_size, cfg.sample_num, data_dir, 'real_withLabel',
            num_img_per_epoch=cfg.num_mini_batch_per_epoch*cfg.real_bs, threshold=cfg.outlier_th)
            
        real_dataloader = torch.utils.data.DataLoader(real_dataset,
            batch_size=cfg.real_bs,
            num_workers=cfg.real_num_workers,
            shuffle=cfg.shuffle,
            sampler=None,
            drop_last=cfg.drop_last,
            pin_memory=cfg.pin_memory)

        data_loader['syn'] = syn_dataloader
        data_loader['real'] = real_dataloader
    
    elif cfg.dataset_name == "camera":
        syn_dataset = TrainingDataset(
            cfg.image_size, cfg.sample_num, data_dir, 'syn',
            num_img_per_epoch=cfg.num_mini_batch_per_epoch*cfg.syn_bs, threshold=cfg.outlier_th)
            
        syn_dataloader = torch.utils.data.DataLoader(syn_dataset,
            batch_size=cfg.syn_bs,
            num_workers=cfg.syn_num_workers,
            shuffle=cfg.shuffle,
            sampler=None,
            drop_last=cfg.drop_last,
            pin_memory=cfg.pin_memory)
            
        data_loader['syn'] = syn_dataloader
    
    elif cfg.dataset_name == "housecat6d":
        real_dataset = HouseCat6DTrainingDataset(
            cfg.image_size, cfg.sample_num, data_dir, cfg.seq_length, cfg.img_length)
        
        real_dataloader = torch.utils.data.DataLoader(real_dataset,
            batch_size=cfg.batchsize,
            num_workers=int(cfg.num_workers),
            shuffle=cfg.shuffle,
            sampler=None,
            drop_last=cfg.drop_last,
            pin_memory=cfg.pin_memory)
        
        data_loader['real'] = real_dataloader
    elif cfg.dataset_name == "COPE119":
        # Get optional path configurations
        train_list_path = getattr(cfg, 'train_list_path', None)
        model_path = getattr(cfg, 'model_path', None)
        pkl_dir = getattr(cfg, 'pkl_dir', None)
        
        finetune_dataset = FinetuneDataset(
            cfg.image_size, cfg.sample_num, data_dir, 'real',
            num_img_per_epoch=cfg.num_mini_batch_per_epoch*cfg.real_bs, threshold=cfg.outlier_th,
            train_list_path=train_list_path, model_path=model_path, pkl_dir=pkl_dir)
            
        finetune_dataloader = torch.utils.data.DataLoader(finetune_dataset,
            batch_size=cfg.real_bs,
            num_workers=cfg.real_num_workers,
            shuffle=cfg.shuffle,
            sampler=None,
            drop_last=cfg.drop_last,
            pin_memory=cfg.pin_memory)
        
        data_loader['real'] = finetune_dataloader
    elif cfg.dataset_name == "REAL275":
        finetune_dataset = FinetuneOnREAL275Dataset(
            cfg.image_size, cfg.sample_num, data_dir, 'real_withLabel',
            num_img_per_epoch=cfg.num_mini_batch_per_epoch*cfg.real_bs, threshold=cfg.outlier_th)
            
        finetune_dataloader = torch.utils.data.DataLoader(finetune_dataset,
            batch_size=cfg.real_bs,
            num_workers=cfg.real_num_workers,
            shuffle=cfg.shuffle,
            sampler=None,
            drop_last=cfg.drop_last,
            pin_memory=cfg.pin_memory)
        
        data_loader['real'] = finetune_dataloader
    else:
        raise NotImplementedError
    
    return data_loader