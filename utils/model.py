import numpy as np
import torch


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def data_to_device(batch, device, mode='train'):
    spker_ids = torch.from_numpy(batch["spker_id"]).to(device)
    text_seq = torch.from_numpy(batch["text_seq"]).long().to(device)
    unk_text = torch.from_numpy(batch["unk_text"]).long().to(device)
    unk_place = torch.from_numpy(batch["unk_place"]).long().to(device)
    unk_id = torch.from_numpy(batch["unk_id"]).long().to(device)
    unk_cls = torch.from_numpy(batch["unk_cls"]).long().to(device)
    text_len = torch.from_numpy(batch["text_len"]).long().to(device)
    unk_text_len = torch.from_numpy(batch["unk_text_len"]).long().to(device)
    d_gt = torch.from_numpy(batch["d"]).long().to(device)
    log_d_gt = torch.from_numpy(batch["log_d"]).float().to(device)
    p_gt = torch.from_numpy(batch["f0"]).float().to(device)
    e_gt = torch.from_numpy(batch["energy"]).float().to(device)
    mel_gt = torch.from_numpy(batch["mel"]).float().to(device)
    mel_len = torch.from_numpy(batch["mel_len"]).long().to(device)
    max_text_len = np.max(batch["text_len"]).astype(np.int32)
    max_unk_len = np.max(batch["unk_text_len"]).astype(np.int32)
    max_mel_len = np.max(batch["mel_len"]).astype(np.int32)

    if mode == 'train':
      model_batch = (
          spker_ids,
          text_seq,
          text_len,
          unk_text,
          unk_text_len,
          unk_place,
          d_gt,
          p_gt,
          e_gt,
          mel_len,
          max_text_len,
          max_mel_len,
          max_unk_len,
          unk_id,
          unk_cls
      )
    else:
      model_batch = (
          spker_ids,
          text_seq,
          text_len,
          None,
          None,
          None,
          None,
          None,
          None,
          mel_len,
          max_text_len,
          max_mel_len,
          None,
          None,
          None
      )

    gt_batch = (log_d_gt, p_gt, e_gt, mel_gt, unk_id, unk_cls, unk_place)
    return (model_batch, gt_batch)
