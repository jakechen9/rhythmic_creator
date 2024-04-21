import pretty_midi


def find_idx(given):
    int_idx = None
    min_float_idx = None
    max_float_idx = None
    for i, entry in enumerate(given[0]):
        if isinstance(entry, int):
            int_idx = i
        elif isinstance(entry, float):
            if min_float_idx is None or entry < given[0][min_float_idx]:
                min_float_idx = i
            if max_float_idx is None or entry > given[0][max_float_idx]:
                max_float_idx = i
    return int_idx, min_float_idx, max_float_idx


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
