"""Training-time caption augmentation (dependency-free).

Tag shuffle, sentence shuffle, tag dropout, full-caption (CFG) dropout, and
tags / natural-language mixing with protected tags. Ported from the Anima
pipeline so any model can reuse it. Pure stdlib + ``random`` — no torch/model
imports — so it's cheap to import from a data pipeline.

These augmentations re-roll the caption *every step*, so they only take effect
when text embeddings are NOT cached (the model must encode on-the-fly). Call
``process_caption`` per sample inside ``prepare_inputs``.
"""

import random
import re
from pathlib import Path

# Minimum number of tags that must survive dropout.
MIN_SURVIVING_TAGS = 3

# Default sampling weights for caption_mode='mixed'.
DEFAULT_MIXED_WEIGHTS = {'tags': 50, 'nl': 10, 'tags_nl': 20, 'nl_tags': 20}

# Default pattern(s) used to recognize an "attribution" entry (e.g. a
# "Drawn by {artist}" tag/sentence) inside a tag list or an NL sentence list.
DEFAULT_ATTRIBUTION_PATTERNS = [r'^drawn by\s']


def build_caption_config(model_config):
    """Extract caption-processing options from a model config dict."""
    return {
        'shuffle_tags': model_config.get('shuffle_tags', False),
        'tag_delimiter': model_config.get('tag_delimiter', ', '),
        'shuffle_keep_first_n': model_config.get('shuffle_keep_first_n', 0),
        'tag_dropout_percent': model_config.get('tag_dropout_percent', 0.0),
        'nl_shuffle_sentences': model_config.get('nl_shuffle_sentences', False),
        'nl_keep_first_sentence': model_config.get('nl_keep_first_sentence', False),
        'caption_dropout_percent': model_config.get('caption_dropout_percent', 0.0),
        'caption_mode': model_config.get('caption_mode', 'tags'),
        'mixed_weights': model_config.get('mixed_weights', DEFAULT_MIXED_WEIGHTS),
        'debug_caption_processing': model_config.get('debug_caption_processing', False),
        'debug_caption_interval': model_config.get('debug_caption_interval', 100),
        # --- Attribution handling (e.g. "Drawn by {artist}") ---
        'attribution_patterns': model_config.get('attribution_patterns', DEFAULT_ATTRIBUTION_PATTERNS),
        'attribution_position': model_config.get('attribution_position', 'fixed'),
        'attribution_dropout_immune': model_config.get('attribution_dropout_immune', True),
        'attribution_dedupe_on_combine': model_config.get('attribution_dedupe_on_combine', True),
    }


def validate_caption_config(config):
    caption_mode = config.get('caption_mode', 'tags')
    valid_modes = ['tags', 'nl', 'mixed']
    if caption_mode not in valid_modes:
        raise ValueError(f"caption_mode must be one of {valid_modes}, got '{caption_mode}'")
    dropout = config.get('tag_dropout_percent', 0.0)
    if not 0.0 <= dropout <= 1.0:
        raise ValueError(f"tag_dropout_percent must be in [0,1], got {dropout}")
    caption_dropout = config.get('caption_dropout_percent', 0.0)
    if not 0.0 <= caption_dropout <= 1.0:
        raise ValueError(f"caption_dropout_percent must be in [0,1], got {caption_dropout}")
    if caption_mode in ['nl', 'mixed']:
        print(f"Note: caption_mode='{caption_mode}' expects {{name}}_nl.txt files. "
              "Samples without NL captions fall back to tags.")
    attribution_position = config.get('attribution_position', 'fixed')
    if attribution_position not in ('fixed', 'random'):
        raise ValueError(f"attribution_position must be 'fixed' or 'random', got '{attribution_position}'")
    try:
        _compile_attribution_patterns(config.get('attribution_patterns'))
    except re.error as e:
        raise ValueError(f"Invalid attribution_patterns regex: {e}")


def caption_config_needs_on_the_fly(config):
    """True if any option re-rolls the caption combinatorially per step and so
    cannot be cached. Tags/NL *mixing* (caption_mode) is NOT here: its variants
    form a finite discrete set that can be cached and weighted-selected per step.
    """
    return bool(
        config.get('shuffle_tags')
        or config.get('tag_dropout_percent')
        or config.get('caption_dropout_percent')
        or config.get('nl_shuffle_sentences')
    )


def load_protected_tags(filepath):
    """Load protected tags (one per line, '#' comments allowed) into a set."""
    if not filepath:
        return set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tags = set()
            for line in f:
                tag = line.strip()
                if tag and not tag.startswith('#'):
                    tags.add(tag)
            return tags
    except FileNotFoundError:
        print(f"Warning: protected_tags_file not found: {filepath}")
        return set()
    except Exception as e:
        print(f"Warning: Error loading protected_tags_file: {e}")
        return set()


def _apply_tag_dropout(tags, dropout_percent, protected_indices, protected_tags):
    """Drop a fraction of tags, keeping protected ones and a minimum count."""
    if dropout_percent <= 0 or len(tags) == 0:
        return tags, []

    droppable_indices = []
    for i, tag in enumerate(tags):
        if i in protected_indices:
            continue
        if tag.strip() in protected_tags:
            continue
        droppable_indices.append(i)

    if len(droppable_indices) == 0:
        return tags, []

    num_to_drop = round(len(droppable_indices) * dropout_percent)
    max_droppable = len(tags) - MIN_SURVIVING_TAGS
    num_to_drop = min(num_to_drop, max(0, max_droppable))
    if num_to_drop == 0:
        return tags, []

    drop_indices = set(random.sample(droppable_indices, num_to_drop))
    surviving, dropped = [], []
    for i, tag in enumerate(tags):
        (dropped if i in drop_indices else surviving).append(tag)
    return surviving, dropped


def _process_nl_caption(nl_caption, shuffle_sentences, keep_first_sentence):
    """Optionally shuffle sentences of an NL caption."""
    if not shuffle_sentences or not nl_caption:
        return nl_caption
    sentences = [s.strip() for s in nl_caption.split('. ') if s.strip()]
    if len(sentences) <= 1:
        return nl_caption
    if keep_first_sentence:
        first, rest = sentences[0], sentences[1:]
        random.shuffle(rest)
        sentences = [first] + rest
    else:
        random.shuffle(sentences)
    result = '. '.join(s.rstrip('.') for s in sentences)
    if not result.endswith('.'):
        result += '.'
    return result


# --- Attribution handling ("Drawn by {artist}" style trigger entries) ---
#
# An "attribution" entry is a single tag (in the tags list) or a single
# sentence (in the NL caption) matching one of `attribution_patterns`, e.g.
# "Drawn by emily (pure dream)". Because the same phrase is typically present
# in *both* the tags string and the NL string (so it survives whichever
# variant training picks), it needs special handling in two places:
#   1. Per-field: it should optionally be immune to shuffle-driven loss and
#      tag dropout (it's a trigger word, not decoration), and its position
#      (pinned to the front vs. free to land anywhere) is a training-strategy
#      choice, not a fixed rule.
#   2. At combine time (tags_nl / nl_tags variants): if it's present in both
#      fields, the non-leading field's copy should usually be dropped so it
#      isn't repeated twice in one training caption.

def _compile_attribution_patterns(patterns):
    return [re.compile(p, re.IGNORECASE) for p in (patterns or DEFAULT_ATTRIBUTION_PATTERNS)]


def _is_attribution(entry, compiled_patterns):
    entry = entry.strip()
    return any(p.search(entry) for p in compiled_patterns)


def _extract_attribution(entries, compiled_patterns):
    """Pull the first entry matching an attribution pattern out of a list.

    Returns (attribution_entry_or_None, remaining_entries). Matches anywhere
    in the list (not just index 0), so this is safe regardless of where the
    attribution currently sits in the raw caption file.
    """
    for i, entry in enumerate(entries):
        if _is_attribution(entry, compiled_patterns):
            return entry, entries[:i] + entries[i + 1:]
    return None, entries


def _reinsert_attribution(entries, attribution, position):
    """Insert `attribution` into `entries` per `position` ('fixed'|'random')."""
    if attribution is None:
        return entries
    if position == 'random':
        idx = random.randint(0, len(entries))
        return entries[:idx] + [attribution] + entries[idx:]
    return [attribution] + entries  # 'fixed'


def _pop_attribution_if_immune(entries, compiled_patterns, immune):
    """Phase 1 (before shuffle/dropout): remove the attribution entry only if
    it's meant to be immune, so shuffle/dropout never see it. If not immune,
    leave it in `entries` untouched so it's treated like any other entry.
    Returns (held_attribution_or_None, entries_to_process).
    """
    if not immune:
        return None, entries
    return _extract_attribution(entries, compiled_patterns)


def _settle_attribution(entries, held_attribution, compiled_patterns, position):
    """Phase 2 (after shuffle/dropout): place the attribution at its final
    position. If it was held out (immune case), reinsert it. If it wasn't
    held out (not-immune case), it may or may not have survived dropout; if
    it survived and position=='fixed', pin it back to the front, otherwise
    leave it wherever shuffle/dropout left it.
    """
    if held_attribution is not None:
        return _reinsert_attribution(entries, held_attribution, position)
    if position == 'fixed':
        found, rest = _extract_attribution(entries, compiled_patterns)
        if found is not None:
            return [found] + rest
    return entries


def _remove_attribution_from_text(text, compiled_patterns, delimiter):
    """Strip any attribution entry out of an already-joined caption fragment.
    `delimiter` is the tag delimiter (e.g. ', ') for tag-style text, or None
    for NL-style text (split on sentences instead).
    """
    if not text:
        return text
    if delimiter is not None:
        entries = [e.strip() for e in text.split(delimiter) if e.strip()]
        entries = [e for e in entries if not _is_attribution(e, compiled_patterns)]
        return delimiter.join(entries)
    sentences = [s.strip() for s in text.split('. ') if s.strip()]
    sentences = [s for s in sentences if not _is_attribution(s, compiled_patterns)]
    if not sentences:
        return ""
    result = '. '.join(s.rstrip('.') for s in sentences)
    if not result.endswith('.'):
        result += '.'
    return result


def _select_variant(caption_mode, mixed_weights, has_nl_caption):
    """Pick a caption variant ('tags' | 'nl' | 'tags_nl' | 'nl_tags')."""
    if caption_mode == "tags":
        return "tags"
    if caption_mode == "nl":
        if has_nl_caption:
            return "nl"
        if not hasattr(_select_variant, '_nl_fallback_count'):
            _select_variant._nl_fallback_count = 0
        _select_variant._nl_fallback_count += 1
        if _select_variant._nl_fallback_count <= 5:
            print("Warning: caption_mode='nl' but no *_nl.txt found for sample, "
                  f"falling back to tags (warning {_select_variant._nl_fallback_count}/5)")
        elif _select_variant._nl_fallback_count == 6:
            print("Warning: Suppressing further NL fallback warnings.")
        return "tags"
    if caption_mode == "mixed":
        available = {"tags": mixed_weights.get("tags", 50)}
        if has_nl_caption:
            available["nl"] = mixed_weights.get("nl", 10)
            available["tags_nl"] = mixed_weights.get("tags_nl", 20)
            available["nl_tags"] = mixed_weights.get("nl_tags", 20)
        total = sum(available.values())
        if total == 0:
            return "tags"
        r = random.random() * total
        cumulative = 0
        for variant, weight in available.items():
            cumulative += weight
            if r < cumulative:
                return variant
        return variant
    return "tags"


def _construct_caption(variant, processed_tags, processed_nl, attribution_patterns=None,
                        dedupe_on_combine=True, tag_delimiter=', '):
    """Combine tag/NL components per the chosen variant, handling empties.

    When combining tags+NL (tags_nl / nl_tags) and `dedupe_on_combine` is on,
    strip any attribution entry out of whichever section is NOT leading, so
    e.g. "Drawn by X" isn't repeated once from the tags side and once from
    the NL side in the same training caption.
    """
    tags = processed_tags.strip() if processed_tags else ""
    nl = processed_nl.strip() if processed_nl else ""

    if dedupe_on_combine and tags and nl and variant in ("tags_nl", "nl_tags"):
        compiled = _compile_attribution_patterns(attribution_patterns)
        if variant == "tags_nl":
            nl = _remove_attribution_from_text(nl, compiled, delimiter=None)
        else:  # nl_tags
            tags = _remove_attribution_from_text(tags, compiled, delimiter=tag_delimiter)

    if variant == "tags":
        return tags if tags else nl
    if variant == "nl":
        return nl if nl else tags
    if variant == "tags_nl":
        return f"{tags}. {nl}" if (tags and nl) else (tags or nl)
    if variant == "nl_tags":
        return f"{nl}. {tags}" if (tags and nl) else (nl or tags)
    return tags or nl


def _load_nl_caption(image_spec):
    """Load '{basename}_nl.txt' next to the image, or None."""
    if image_spec is None:
        return None
    tar_file, image_path = image_spec
    if tar_file is not None or not image_path:
        return None
    image_path = Path(image_path)
    nl_path = image_path.parent / f"{image_path.stem}_nl.txt"
    if not nl_path.exists():
        return None
    try:
        with open(nl_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return content or None
    except Exception:
        return None


def _should_debug_sample(sample_idx, interval):
    if interval == 0:
        return True
    if interval == -1:
        return sample_idx < 10
    return sample_idx % interval == 0


def _print_debug(sample_idx, info, full_dropout):
    print(f"\n[Caption Debug | Sample {sample_idx}]")
    if full_dropout:
        print("├─ Full caption dropout: YES (CFG training)")
        print("├─ Final caption: \"\"")
        print("└─ (all other processing skipped)")
        return
    print(f"├─ Original tags: \"{info.get('original_tags', '')}\"")
    nl = info.get('original_nl')
    print(f"├─ Original NL: \"{nl or '(none)'}\"")
    if info.get('dropped_tags'):
        print(f"├─ Dropped tags: {info['dropped_tags']}")
    print(f"├─ Surviving tags: \"{info.get('surviving_tags', '')}\"")
    if info.get('processed_nl'):
        print(f"├─ Processed NL (post shuffle/attribution): \"{info['processed_nl']}\"")
    print(f"├─ Variant selected: {info.get('variant', 'unknown')}")
    final = info.get('final_caption', '')
    print(f"└─ Final caption: \"{final}\"")


def log_caption_stats(debug_state, step, interval=1000):
    if step % interval != 0 or step == 0:
        return
    variants = ['tags', 'nl', 'tags_nl', 'nl_tags']
    counts = [debug_state.get(f'variant_{v}', 0) for v in variants]
    total = sum(counts)
    if total == 0:
        return
    pcts = [f"{v}={c}({100 * c // total}%)" for v, c in zip(variants, counts)]
    print(f"Step {step} | Variants: {', '.join(pcts)} | "
          f"Tag drops: {debug_state.get('tag_dropout_count', 0)} | "
          f"CFG drops: {debug_state.get('full_dropout_count', 0)}")


def process_caption(tags_str, image_spec, config, protected_tags, sample_idx, debug_state):
    """Full per-sample caption pipeline. Returns the final caption string.

    Steps: full-caption (CFG) dropout -> load NL if needed -> tag shuffle ->
    tag dropout -> NL sentence shuffle -> variant selection -> construct.
    """
    debug_info = {}
    debug_enabled = config.get('debug_caption_processing', False)
    debug_interval = config.get('debug_caption_interval', 100)
    should_debug = debug_enabled and _should_debug_sample(sample_idx, debug_interval)

    # Step 1: full caption dropout (unconditional / CFG training).
    caption_dropout = config.get('caption_dropout_percent', 0.0)
    if caption_dropout > 0 and random.random() < caption_dropout:
        debug_state['full_dropout_count'] = debug_state.get('full_dropout_count', 0) + 1
        if should_debug:
            _print_debug(sample_idx, debug_info, full_dropout=True)
        return ""

    if should_debug:
        debug_info['original_tags'] = tags_str

    # Step 2: load NL caption if the mode uses it.
    caption_mode = config.get('caption_mode', 'tags')
    nl_caption = _load_nl_caption(image_spec) if caption_mode in ['nl', 'mixed'] else None
    if should_debug:
        debug_info['original_nl'] = nl_caption

    # Attribution policy setup (shared by tags and NL below).
    attribution_patterns = _compile_attribution_patterns(config.get('attribution_patterns'))
    attribution_immune = config.get('attribution_dropout_immune', True)
    attribution_position = config.get('attribution_position', 'fixed')

    # Step 3: parse + shuffle + drop tags.
    delimiter = config.get('tag_delimiter', ', ')
    tags = [t.strip() for t in tags_str.split(delimiter) if t.strip()]

    tag_attribution, tags = _pop_attribution_if_immune(tags, attribution_patterns, attribution_immune)

    if config.get('shuffle_tags', False):
        keep_first_n = config.get('shuffle_keep_first_n', 0)
        if 0 < keep_first_n < len(tags):
            prefix, suffix = tags[:keep_first_n], tags[keep_first_n:]
            random.shuffle(suffix)
            tags = prefix + suffix
        else:
            random.shuffle(tags)

    dropout_percent = config.get('tag_dropout_percent', 0.0)
    keep_first_n = config.get('shuffle_keep_first_n', 0)
    protected_indices = set(range(min(keep_first_n, len(tags))))
    dropped_tags = []
    if dropout_percent > 0:
        tags, dropped_tags = _apply_tag_dropout(tags, dropout_percent, protected_indices, protected_tags)
        debug_state['tag_dropout_count'] = debug_state.get('tag_dropout_count', 0) + len(dropped_tags)

    tags = _settle_attribution(tags, tag_attribution, attribution_patterns, attribution_position)
    processed_tags = delimiter.join(tags)

    if should_debug:
        debug_info['dropped_tags'] = dropped_tags
        debug_info['surviving_tags'] = processed_tags

    # Step 4: process NL caption (sentence shuffle + attribution policy).
    has_nl = bool(nl_caption and nl_caption.strip())
    processed_nl = ""
    if has_nl:
        sentences = [s.strip() for s in nl_caption.split('. ') if s.strip()]
        nl_attribution, sentences = _pop_attribution_if_immune(sentences, attribution_patterns, attribution_immune)

        if config.get('nl_shuffle_sentences', False) and len(sentences) > 1:
            if config.get('nl_keep_first_sentence', False):
                first, rest = sentences[0], sentences[1:]
                random.shuffle(rest)
                sentences = [first] + rest
            else:
                random.shuffle(sentences)

        sentences = _settle_attribution(sentences, nl_attribution, attribution_patterns, attribution_position)
        processed_nl = '. '.join(s.rstrip('.') for s in sentences)
        if processed_nl and not processed_nl.endswith('.'):
            processed_nl += '.'

    if should_debug:
        debug_info['processed_nl'] = processed_nl

    # Step 5: pick a variant and construct.
    mixed_weights = config.get('mixed_weights', DEFAULT_MIXED_WEIGHTS)
    variant = _select_variant(caption_mode, mixed_weights, has_nl)
    debug_state[f'variant_{variant}'] = debug_state.get(f'variant_{variant}', 0) + 1
    if should_debug:
        debug_info['variant'] = variant

    final_caption = _construct_caption(
        variant, processed_tags, processed_nl,
        attribution_patterns=config.get('attribution_patterns'),
        dedupe_on_combine=config.get('attribution_dedupe_on_combine', True),
        tag_delimiter=delimiter,
    )

    # Step 6: never return empty (unless it was intentional CFG dropout above).
    if not final_caption or not final_caption.strip():
        final_caption = tags_str

    if should_debug:
        debug_info['final_caption'] = final_caption
        _print_debug(sample_idx, debug_info, full_dropout=False)
    return final_caption