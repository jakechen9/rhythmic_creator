import pretty_midi
import torch
from preprocess.preprocessing import MIDIProcessor


# extract given context
class Extract(MIDIProcessor):
    def __init__(self, path, input_midi_path):
        super().__init__(path)
        self.path = path
        self.input_midi_path = input_midi_path

    def extract_drum_pattern(self, device):
        dat = pretty_midi.PrettyMIDI(self.input_midi_path)
        drum_notes = []
        for instrument in dat.instruments:
            for note in instrument.notes:
                drum_notes.append([str(int(note.pitch)), str(round(note.start, 2)), str(round(note.end, 2))])
        flat_list = [item for sublist in drum_notes for item in sublist]
        context_tensor = torch.tensor(MIDIProcessor(self.path).encode_with_mapping(flat_list),
                                      dtype=torch.long, device=device)
        context = torch.reshape(context_tensor, (1, len(flat_list)))
        return context
