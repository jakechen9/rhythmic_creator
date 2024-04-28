import torch
from preprocess import preprocessing
from models import transformerdecoder, lstm_integration
from utils import extract, generate

path = 'training_1.txt'
MIDIPreprocess = preprocessing.MIDIProcessor(path)
midi_flatten, unique_notes = MIDIPreprocess.split_text()
data = torch.tensor(MIDIPreprocess.encode_with_mapping(midi_flatten), dtype=torch.long)

batch_size = 64
block_size = 256
vocab_size = len(unique_notes)
n_embd = 192
num_heads = 6  # 192/6 = 32, which is dim, and our head_size
n_layer = 6
dropout = 0.2
n_hidden = 64
lstm_layers = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

while True:
    model_type = input("Choose the model from 'Transformer_Hybrid', 'Transformer': ")
    if model_type in ['Transformer_Hybrid', 'Transformer']:
        break
    else:
        print("Error: Invalid input. Please enter either 'Transformer_Hybrid' or 'Transformer'.")

while True:
    conditional = input("Conditional yes or no: ")
    if conditional.lower() in ['yes', 'no']:
        break
    else:
        print("Error: Invalid input. Please enter 'yes' or 'no'.")

if conditional.lower() == 'yes':
    while True:
        context_path = input("Enter path to context: ")
        if context_path.endswith(('.mid', '.midi')):
            break
        else:
            print("Error: Invalid input. The path must end with either '.mid' or '.midi'.")
else:
    context_path = None

while True:
    number_notes = input("Num of notes to generate: ")
    if number_notes.isdigit() and int(number_notes) > 0:
        break
    else:
        print("Error: Invalid input. Please enter positive integers.")


def get_max_tok_and_context(conditional_type):
    max_tok = 0
    given = None
    if conditional == 'yes':
        context = extract.Extract('training_1.txt', context_path)
        given = context.extract_drum_pattern(device)
        max_tok = int(number_notes) * 3
    elif conditional == 'no':
        max_tok = int(number_notes) * 3 - 1
        given = torch.zeros((1, 1), dtype=torch.long, device=device)
    return max_tok, given


max_token, given_context = get_max_tok_and_context(conditional)

while True:
    gen_save_path = input("Enter path to save the file: ")
    if gen_save_path.endswith(('.mid', '.midi')):
        break
    else:
        print("Error: Invalid input. The path must end with either '.mid' or '.midi'.")


def generate_midi(model):
    if model == 'Transformer':
        m = transformerdecoder.DecoderModel(block_size, vocab_size, n_embd, num_heads, n_layer, dropout).to(device)
        m.load_state_dict(torch.load('/Users/cheng/thesis/transformer_base_192d.pt', map_location=torch.device('cpu')))
        m.eval()
        given_generated = m.generate(device, given_context, max_new_tokens=max_token)
        given_decode = MIDIPreprocess.decode_with_mapping(given_generated[0].tolist()).split()
        gen = [given_decode[i:i + 3] for i in range(0, len(given_decode), 3)]
        return gen

    elif model == 'Transformer_Hybrid':
        m = lstm_integration.LSTMDecoderModel(block_size, vocab_size, n_embd, num_heads, n_layer, dropout, n_hidden,
                                              lstm_layers).to(device)
        m.load_state_dict(
            torch.load('/Users/cheng/thesis/transformer_LSTM_FNN_hybrid.pt', map_location=torch.device('cpu')))
        m.eval()
        hidden = m.init_hidden(1, device)
        given_generated = m.generate(device, given_context, hidden, max_new_tokens=max_token)
        given_decode = MIDIPreprocess.decode_with_mapping(given_generated[0].tolist()).split()
        gen = [given_decode[i:i + 3] for i in range(0, len(given_decode), 3)]
        return gen


generated = generate_midi(model_type)
generate.gen(generated, gen_save_path)
