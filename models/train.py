import torch
from preprocess.batch import Batch
from preprocess.preprocessing import MIDIProcessor
from models.transformerdecoder import DecoderModel


path = '/Users/cheng/thesis/rhythmic_creator/training_1.txt'
MIDIPreprocess = MIDIProcessor(path)
midi_flatten, unique_notes = MIDIPreprocess.split_text()
data = torch.tensor(MIDIPreprocess.encode_with_mapping(midi_flatten), dtype=torch.long)
print(data[:9])


# model = DecoderModel(block_size, vocab_size, n_embd, num_heads, n_layer, dropout)
# m = model.to(device)
#
# @torch.no_grad()
# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = getbatch.create_batch(split, device)
#             logits, loss = model(device, X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out
