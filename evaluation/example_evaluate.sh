# Example evaluation script

# Unconditional model evaluation
python evaluation/evaluate.py \
    --checkpoint /Users/wenhan/Downloads/tap_ckpt/uc_all/last.ckpt \
    --dataset /Users/wenhan/Downloads/data/unconditional_all \
    --output_dir results/evaluation \
    --feature unconditional \
    --device cpu \
    --temperatures  0.4 0.5 0.6 0.7 \
    --split test


# Predict on a single MIDI file
python evaluation/predict_single_midi.py \
    --checkpoint checkpoints/model.ckpt \
    --input_midi input.mid \
    --output_midi output.mid \
    --feature unconditional \
    --device cuda \
    --temperature 1.0