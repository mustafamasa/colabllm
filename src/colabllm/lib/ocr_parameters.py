"""Public inference settings for the pinned Unlimited-OCR 2 Transformers model.

Only tokenizer, image_file(s), and output_path are owned by the job runner.
"""
import argparse
import json
import math

# name -> (JSON type, minimum, nullable). No upper caps beyond the model itself.
PARAMETERS = {
    'prompt': (str, None, False),
    'base_size': (int, 1, False),
    'image_size': (int, 1, False),
    'crop_mode': (bool, None, True),
    'test_compress': (bool, None, False),
    'save_results': (bool, None, False),
    'eval_mode': (bool, None, False),
    'max_length': (int, 1, False),
    'max_new_tokens': (int, 1, True),
    'tps_interval': (int, 0, False),
    'no_repeat_ngram_size': (int, 0, False),
    'ngram_window': (int, 0, False),
    'ngram_whitelist_token_ids': (list, None, True),
    'temperature': (float, 0, False),
    'page_count_guard': (bool, None, False),
    'max_page_tokens': (int, None, False),
    'dash_topm': (int, 1, False),
    'max_crops': (int, 1, False),
    'return_dict': (bool, None, False),
}


def validate_parameters(value):
    if not isinstance(value, dict):
        raise ValueError('inference_params must be a JSON object')
    for name, item in value.items():
        if name not in PARAMETERS:
            raise ValueError(f'Unsupported inference parameter: {name}; available: {", ".join(PARAMETERS)}')
        kind, minimum, nullable = PARAMETERS[name]
        if item is None and nullable:
            continue
        valid = type(item) is kind
        if kind is float:
            valid = type(item) in (int, float) and math.isfinite(item)
        if not valid or (minimum is not None and item < minimum):
            raise ValueError(f'{name}: type must be {kind.__name__}' + (f', value >= {minimum}' if minimum is not None else ''))
        if kind is list and any(type(n) is not int or n < 0 for n in item):
            raise ValueError(f'{name} must be an array of non-negative integers or null')
    return dict(value)


def inference_kwargs(params, page_count, tokens_per_page):
    """Keep historical service defaults; explicit values, including null, always win."""
    result = dict(eval_mode=True, crop_mode=False, return_dict=True,
                  max_new_tokens=min(120000, tokens_per_page * page_count),
                  no_repeat_ngram_size=35, ngram_window=1024 if page_count > 1 else 128)
    result.update(validate_parameters(params))
    return result


def add_inference_arguments(parser):
    parser.add_argument('--params-file', help='local JSON file with inference parameters (official parameter names)')
    parser.add_argument('--params-json', help='JSON object with inference parameters; overrides the parameter file')
    parser.add_argument('--param', action='append', default=[], metavar='KEY=JSON',
                        help='repeatable; overrides the JSON configuration, e.g. --param crop_mode=null')
    for name, (kind, _, _) in PARAMETERS.items():
        flag = '--' + name.replace('_', '-')
        if kind is bool:
            parser.add_argument(flag, dest=name, action=argparse.BooleanOptionalAction, default=None)
        elif kind is list:
            parser.add_argument(flag, dest=name, type=json.loads, default=None, metavar='JSON_ARRAY')
        else:
            parser.add_argument(flag, dest=name, type=kind, default=None)


def cli_parameters(args):
    from pathlib import Path
    result = {}
    for raw in ([Path(args.params_file).expanduser().read_text(encoding='utf-8')] if args.params_file else []) + ([args.params_json] if args.params_json else []):
        result.update(validate_parameters(json.loads(raw)))
    for name in PARAMETERS:
        value = getattr(args, name)
        if value is not None:
            result[name] = value
    for setting in args.param:
        name, separator, raw = setting.partition('=')
        if not separator:
            raise ValueError('--param must have the form KEY=JSON')
        result[name] = json.loads(raw)
    return validate_parameters(result)
