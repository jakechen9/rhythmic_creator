import pretty_midi


def find_idx(given):
    int_idx = None
    float_val = []
    for i, entry in enumerate(given[0]):
        try:
            int(entry)
            int_idx = i
        except ValueError:
            float_val.append(float(entry))
    if float_val:
        min_idx = float_val.index(min(float_val))
        max_idx = float_val.index(max(float_val))
    else:
        min_idx = None
        max_idx = None
    return int_idx, min_idx, max_idx


def gen(given, output_path):
    input_tempo = 120.0
    drum_pattern = pretty_midi.PrettyMIDI(initial_tempo=input_tempo)
    drum_program = pretty_midi.instrument_name_to_program('cello')
    drum = pretty_midi.Instrument(program=drum_program)
    pitch_idx, start_idx, end_idx = find_idx(given)
    for i in range(len(given)):
        note = pretty_midi.Note(velocity=100, pitch=int(given[i][pitch_idx]), start=float(given[i][start_idx]),
                                end=float(given[i][end_idx]))
        drum.notes.append(note)
    drum_pattern.instruments.append(drum)
    drum_pattern.write(output_path)
