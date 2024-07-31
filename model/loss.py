import torch
import torch.nn as nn
from config import hparams as hp


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self):
        super(FastSpeech2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.unk_ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([1, 10]).float())

    def forward(
        self,
        mel,
        mel_postnet,
        log_d_predicted,
        p_predicted,
        e_predicted,
        unk_predicted,
        unk_place_predict,
        log_d_target,
        f0_gt,
        e_target,
        mel_target,
        unk_id,
        unk_cls,
        unk_place,
        src_mask,
        mel_mask,
    ):
        """
        all input will be flattened before mse or mae
        """
        log_d_target.requires_grad = False
        f0_gt.requires_grad = False
        e_target.requires_grad = False
        mel_target.requires_grad = False
        unk_id.requires_grad = False
        unk_cls.requires_grad = False
        unk_place.requires_grad = False

        log_d_predicted = log_d_predicted.masked_select(src_mask)
        log_d_target = log_d_target.masked_select(src_mask)
        p_predicted = p_predicted.masked_select(src_mask)
        f0_gt = f0_gt.masked_select(src_mask)
        e_predicted = e_predicted.masked_select(src_mask)
        e_target = e_target.masked_select(src_mask)

        # 2021/11/21 we do the mel masking in FastSpeech2 model
        # It's possible to do masking for d, p, e in the model too.
        # mel = mel.masked_select(mel_mask.unsqueeze(-1))
        # mel_postnet = mel_postnet.masked_select(mel_mask.unsqueeze(-1))
        # mel_target = mel_target.masked_select(mel_mask.unsqueeze(-1))

        mel_loss = self.mse_loss(mel, mel_target)
        mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)

        d_loss = self.mae_loss(log_d_predicted, log_d_target)
        p_loss = self.mae_loss(p_predicted, f0_gt)
        e_loss = self.mae_loss(e_predicted, e_target)

        # print(unk_predicted.shape)
        unk_loss = 0
        for i, (predicted, cls) in enumerate(zip(unk_predicted, unk_cls)):
          # print(predicted.shape)
          # print(predicted[unk_id[i].nonzero()].shape)
          # print(cls[unk_id[i].nonzero()])
          # print(predicted[unk_id[i].nonzero()].squeeze())
          # print(cls[unk_id[i].nonzero()].squeeze())
          # print(predicted[unk_id[i].nonzero()].shape, cls[unk_id[i].nonzero()].shape)
          # print(predicted[unk_id[i].nonzero()].squeeze().shape, cls[unk_id[i].nonzero()].squeeze(1).shape)
          unk_loss += self.ce_loss(predicted[unk_id[i].nonzero()].squeeze(1), cls[unk_id[i].nonzero()].squeeze(1))
        unk_loss /= unk_predicted.shape[0]  
        unk_place_loss = sum([self.unk_ce_loss(predict, target) for predict, target in zip(unk_place_predict, unk_place)]) / unk_place_predict.shape[0]

        return (
            mel_loss,
            mel_postnet_loss,
            d_loss,
            0.1 * p_loss,
            0.1 * e_loss,
            unk_loss,
            unk_place_loss
        )
