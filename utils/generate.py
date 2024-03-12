import pretty_midi


def gen(given, output_path):
    input_tempo = 120.0
    drum_pattern = pretty_midi.PrettyMIDI(initial_tempo=input_tempo)
    drum_program = pretty_midi.instrument_name_to_program('cello')
    drum = pretty_midi.Instrument(program=drum_program)
    for i in range(len(given)):
        note = pretty_midi.Note(velocity=100, pitch=int(given[i][0]), start=float(given[i][1]),
                                end=float(given[i][2]))
        drum.notes.append(note)
    drum_pattern.instruments.append(drum)
    drum_pattern.write(output_path)
