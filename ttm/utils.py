"""
Author: Lynn Ye
Created on: 2025/11/12
Brief: 
"""
import logging
import sys
import mido
import numpy as np
from ttm.config import LOG_LEVEL


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[90m[Debug]\033[0m",
        logging.INFO: "\033[34m[Info]\033[0m",
        logging.WARNING: "\033[33m[Warning]\033[0m",
        logging.ERROR: "\033[31m[Error]\033[0m",
        logging.CRITICAL: "\033[1;31m[CRITICAL]\033[0m",
    }

    def format(self, record):
        level_color = self.COLORS.get(record.levelno, "")
        log_fmt = f"{level_color} %(asctime)s | %(message)s"
        formatter = logging.Formatter(log_fmt, "%H:%M:%S")
        return formatter.format(record)


def formatargs(*args):
    """Join multiple args into a space-separated string."""
    return " ".join(map(str, args))


class Log:
    def __init__(self, name="app", level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Only add handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(ColorFormatter())
            self.logger.addHandler(handler)

    def _log(self, level, *args):
        msg = formatargs(*args)
        self.logger.log(level, msg)

    def set_level(self, level): self.logger.setLevel(level)

    def debug(self, *args): self._log(logging.DEBUG, *args)

    def info(self, *args): self._log(logging.INFO, *args)

    def warn(self, *args): self._log(logging.WARNING, *args)

    def error(self, *args): self._log(logging.ERROR, *args)

    def crit(self, *args): self._log(logging.CRITICAL, *args)


clog = Log("color_log")


def note_seq_to_midi(note_array, ticks_per_beat=480, tempo=500000):
    note_array = np.array(note_array, dtype=float)
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Helper: convert seconds → MIDI ticks
    def sec_to_ticks(seconds):
        beats = (seconds * 1e6) / tempo
        return int(round(beats * ticks_per_beat))

    # Sort by onset time
    note_array = note_array[note_array[:, 1].argsort()]

    # Build events
    events = []
    for pitch, onset, duration, velocity in note_array:
        start_tick = sec_to_ticks(onset)
        end_tick = sec_to_ticks(onset + duration)
        events.append((start_tick, 'note_on', int(pitch), int(velocity)))
        events.append((end_tick, 'note_off', int(pitch), 0))

    # Sort events by tick
    events.sort(key=lambda e: e[0])

    # Write events with delta times
    last_tick = 0
    for tick, event_type, pitch, velocity in events:
        delta = tick - last_tick
        track.append(mido.Message(event_type, note=pitch, velocity=velocity, time=delta))
        last_tick = tick
    return mid


def main():
    clog.info('hello', 'world', 1 + 1)
    clog.set_level(LOG_LEVEL)
    clog.debug('this is a debug message')


if __name__ == "__main__":
    main()
