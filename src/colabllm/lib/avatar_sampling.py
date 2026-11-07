"""Validated sampling controls shared by the Avatar CLI and GPU worker."""
import math


def sampling_settings(mode='distilled', steps=None, text_cfg=None, audio_cfg=None):
    if mode not in ('distilled', 'standard'):
        raise ValueError('sampling_mode must be distilled or standard')
    defaults = (8, 1.0, 1.0) if mode == 'distilled' else (50, 4.0, 3.0)
    steps = defaults[0] if steps is None else steps
    text_cfg = defaults[1] if text_cfg is None else text_cfg
    audio_cfg = defaults[2] if audio_cfg is None else audio_cfg
    if isinstance(steps, bool) or not isinstance(steps, (int, float)) or not math.isfinite(steps) or int(steps) != steps or not 8 <= steps <= 100:
        raise ValueError('steps must be an integer from 8 to 100')
    for name, value in (('text_guidance_scale', text_cfg), ('audio_guidance_scale', audio_cfg)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 1 <= value <= 10:
            raise ValueError(name + ' must be a finite number from 1 to 10')
    if mode == 'distilled' and (steps, text_cfg, audio_cfg) != defaults:
        raise ValueError('8-step distilled mode fixes steps=8 and text/audio CFG=1; use standard mode to adjust CFG')
    return dict(sampling_mode=mode, steps=int(steps), text_guidance_scale=float(text_cfg),
                audio_guidance_scale=float(audio_cfg), use_distill=mode == 'distilled')


def apply_sampling_mode(pipe, settings):
    """Called under the worker lock, before generation or checkpoint resume."""
    if settings['use_distill']:
        pipe.dit.enable_loras(['dmd'])
    else:
        pipe.dit.disable_all_loras()


def plan_segments(audio_seconds, frames=93, segments=1, auto_duration=True):
    if not auto_duration:
        total = frames + (segments - 1) * (frames - 13)
        return [frames] * segments, total, total / 25
    if not math.isfinite(audio_seconds) or audio_seconds <= 0:
        raise ValueError('Invalid audio duration')
    target = max(1, math.ceil(round(audio_seconds * 25, 8)))
    # Each model call needs 4n+1 frames; subsequent calls include 13
    # overlapping conditioning frames. Shorten the final call and trim padding.
    parts = [max(17, 1 + 4 * math.ceil((min(target, 93) - 1) / 4))]
    covered = parts[0]
    while covered < target:
        count = min(93, max(17, 13 + 4 * math.ceil((target - covered) / 4)))
        parts.append(count)
        covered += count - 13
        if len(parts) > 100:
            raise ValueError('Audio is too long; split it into clips of at most 5 minutes before generating')
    return parts, target, audio_seconds
