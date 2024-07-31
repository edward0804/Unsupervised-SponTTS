import torch
import torch.nn as nn
import numpy as np
from config import hparams as hp
from transformer.Layers import PostNet
from transformer.Models import Decoder, Encoder
from utils.mask import get_mask_from_lengths
from text.symbols import symbols
from text import _symbol_to_id

from utils.pad import pad_1D

from model.modules import Embedding, SpeakerIntegrator, VarianceAdaptor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unk_index = _symbol_to_id['@unk']

def soft_arg_max(seq):
    index_tensor = torch.Tensor([i for i in range(seq.shape[-1])]).to(device)
    # print('index', index_tensor.shape)
    seq = seq.exp() / seq.exp().sum(dim=-1).unsqueeze(dim=-1) * index_tensor
    # print('1', seq.shape)
    seq = seq.sum(dim=-1) + 1
    seq = seq.long()
    return seq

def process_unk(text_seq, unk_place_predict):
    # print('process_unk input ', text_seq.shape, unk_place_predict.shape)
    unk_texts = []
    unk_places = unk_place_predict.argmax(axis=2).cpu().detach().numpy()
    # print(unk_place_predict)
    # print(unk_places)
    text_seq = text_seq.cpu().detach()
    for text, unk_place in zip(text_seq, unk_places):
      unk_text = []
      # print(text, unk_place)
      for t, p in zip(text, unk_place):
        unk_text.append(t.item())
        if p == 1:
          unk_text.append(unk_index)
      # print(unk_text)
      unk_text = np.array(unk_text)
      unk_texts.append(unk_text)
    unk_text_lens = np.array([text.shape[0] for text in unk_texts])
    unk_texts = pad_1D(unk_texts)
    max_unk_len = np.max(unk_text_lens).astype(np.int32)
    unk_texts = torch.from_numpy(unk_texts).long().to(device)
    unk_text_lens = torch.from_numpy(unk_text_lens).long().to(device)
    # print('unk_place_predict output ', unk_texts.shape, unk_text_lens, max_unk_len)
    return unk_texts, unk_text_lens, max_unk_len



class FastSpeech2(nn.Module):
    """ FastSpeech2 module"""

    def __init__(self, n_spkers=1, spker_embed_dim=256, spker_embed_std=0.01, unk_cls=100):
        super(FastSpeech2, self).__init__()
        self.n_spkers = n_spkers
        self.spker_embed_dim = spker_embed_dim
        self.spker_embed_std = spker_embed_std

        ### Encoder, Speaker Integrator, Variance Adaptor, Deocder, Postnet ###
        self.spker_embeds = Embedding(
            n_spkers, spker_embed_dim, padding_idx=None, std=spker_embed_std
        )
        self.encoder = Encoder(
          n_src_vocab=len(symbols) + unk_cls + 1,
        )
        self.unk_predicter = Encoder(
          n_src_vocab=len(symbols) + 1,
          d_word_vec=2,
          d_k=1,
          d_v=1,
          d_model=2,
          n_layers=1,
          n_head=1,
          d_inner=256,
        )
        self.unk_classifier = Encoder(
          n_src_vocab=len(symbols) + 1,
          d_word_vec=unk_cls,
          d_k=unk_cls // hp.encoder_head,
          d_v=unk_cls // hp.encoder_head,
          d_model=unk_cls,
        )
        self.speaker_integrator = SpeakerIntegrator()
        self.variance_adaptor = VarianceAdaptor()
        self.decoder = Decoder()
        self.to_mel = nn.Linear(hp.decoder_hidden, hp.n_mels)
        self.postnet = PostNet()

    def forward(
        self,
        spker_ids,
        text_seq,
        text_len,
        unk_text=None,
        unk_text_len=None,
        unk_place=None,
        d_gt=None,
        p_gt=None,
        e_gt=None,
        mel_len=None,
        max_text_len=None,
        max_mel_len=None,
        max_unk_len=None,
        unk_id=None,
        unk_cls=None
    ):
        # print('id', unk_id.shape)
        # print('cls', unk_cls.shape)
        # exit()
        text_mask = get_mask_from_lengths(text_len, max_text_len)
        mel_mask = (
            get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None
        )
        # === Encoder === #
        spker_embed = self.spker_embeds(spker_ids)

        # print(text_seq.shape)

        unk_place_predict = self.unk_predicter(text_seq, text_mask)
        # print(f'predict: {unk_place_predict.argmax(axis=2).sum()} / {text_len.sum()}, ans: {unk_place.sum()} / {text_len.sum()}')

        # print('f ', self.training())
        # exit()

        # if not self.training or unk_text == None:
        if unk_text == None:
          print(f'predict: {unk_place_predict.argmax(axis=2).sum()} / {text_len.sum()}, ans: {unk_place.sum()} / {text_len.sum()}')
          unk_text, unk_text_len, max_unk_len = process_unk(text_seq, unk_place_predict)
          print(unk_text_len, text_len)
          # print(unk_text.shape)
  
        unk_text_mask = (
            get_mask_from_lengths(unk_text_len, max_unk_len) if unk_text_len is not None else None
        )
        
        # print('f')
        # exit()

        # print(unk_text.shape, unk_text_mask.shape)
        unk_predict = self.unk_classifier(unk_text, unk_text_mask)
        # print(unk_predict.shape)
        unk_tok = unk_predict.argmax(axis=-1) + len(symbols) + 1

        # if unk_id == None or not self.training:
        if unk_id == None:
          unk_tok_place = (unk_text == unk_index) * 1
          # print(unk_tok.shape, unk_text.shape, unk_tok_place.shape)
          unk_text = (unk_tok) * unk_tok_place + (unk_text * (unk_tok_place == 0))
        else:
          unk_text = (unk_cls) + (unk_text * (unk_id == 0))

        # print('unk_text', unk_text.shape)
        # print('unk_text_mask', unk_text_mask.shape)
        encoder_output = self.encoder(unk_text, unk_text_mask)
        encoder_output = self.speaker_integrator(encoder_output, spker_embed)

        # === Variance Adaptor === #
        if d_gt is not None:
            (
                variance_adaptor_output,
                d_pred,
                p_pred,
                e_pred,
                _,
                _,
            ) = self.variance_adaptor(
                encoder_output, unk_text_mask, mel_mask, d_gt, p_gt, e_gt, max_mel_len,
            )
        else:
            (
                variance_adaptor_output,
                d_pred,
                p_pred,
                e_pred,
                mel_len,
                mel_mask,
            ) = self.variance_adaptor(
                encoder_output, unk_text_mask, mel_mask, d_gt, p_gt, e_gt, max_mel_len,
            )

        variance_adaptor_output = self.speaker_integrator(
            variance_adaptor_output, spker_embed
        )

        # === Decoder === #
        decoder_output = self.decoder(variance_adaptor_output, mel_mask)
        mel = self.to_mel(decoder_output)
        mel_postnet = self.postnet(mel) + mel

        # === Masking === #
        if mel_mask is not None:
            mel = mel.masked_fill(mel_mask.unsqueeze(-1), 0)
            mel_postnet = mel_postnet.masked_fill(mel_mask.unsqueeze(-1), 0)

        # === Output === #
        # print(d_pred.shape, unk_text_mask.shape)
        pred = (mel, mel_postnet, d_pred, p_pred, e_pred, unk_predict, unk_place_predict)
        return (pred, unk_text_mask, mel_mask, mel_len)


if __name__ == "__main__":
    """
    write some tests here
    """
    model = FastSpeech2()
    #print(model)
    #print(sum(param.numel() for param in model.parameters()))
