# Attribution handling in mixed caption mode

When `caption_mode = 'mixed'` (or `'nl'`) is used with `_nl.txt` sidecar
captions, it's common to repeat a style/character trigger phrase — e.g.
`Drawn by emily (pure dream)` — in **both** the tags string and the NL
caption, so the trigger survives whichever variant training happens to pick
for a given step:

```
tags: Drawn by emily (pure dream), 1girl, blonde hair, blue eyes, apron
_nl.txt: Drawn by emily (pure dream). The girl stands in a small room. ...
```

This creates two problems that the options below solve:

1. **Duplication on combine.** `caption_mode='mixed'` can pick a `tags_nl` or
   `nl_tags` variant, which concatenates both fields. Without special
   handling, the attribution phrase would appear twice in a single training
   caption.
2. **Accidental loss.** If the tag list also has `tag_dropout_percent` or
   `shuffle_tags` set, a trigger phrase living as a normal tag can get
   shuffled out of a fixed "first tag" position, or dropped outright by
   dropout — usually not what you want for a trigger word.

## Config options

All are optional; defaults reproduce the common case (fixed position, immune
to dropout, deduped on combine).

```toml
attribution_patterns = ['^drawn by\s']   # regex list, case-insensitive
attribution_position = 'fixed'            # 'fixed' | 'random'
attribution_dropout_immune = true
attribution_dedupe_on_combine = true
```

- **`attribution_patterns`** — list of regexes used to recognize an
  attribution entry. Matched against each tag (after splitting on
  `tag_delimiter`) and each NL sentence (after splitting on `. `),
  independent of where it sits in the list. Default matches `Drawn by ...`.
  Extend this list if you use a different phrasing or want to also treat
  something like a fixed character-name sentence as "attribution".

- **`attribution_position`** — `'fixed'` pins the attribution to the front
  of whichever field it's in (tags or NL), regardless of `shuffle_tags` /
  `nl_shuffle_sentences`. `'random'` lets it land anywhere in that field
  after shuffle — still exactly one copy per field, just not pinned.

- **`attribution_dropout_immune`** — `true` pulls the attribution out
  *before* `tag_dropout_percent` runs, so it can never be dropped. `false`
  leaves it in the normal tag/sentence pool, where it's shuffled and
  dropped like anything else (protect it via `protected_tags_file` instead,
  if you want it undroppable in tags mode specifically but still want this
  flag off for some other reason).

- **`attribution_dedupe_on_combine`** — when a `tags_nl` or `nl_tags`
  variant is selected and the attribution is present in both fields,
  `true` keeps only the copy in the *leading* section (tags first in
  `tags_nl`, NL first in `nl_tags`) and strips it from the trailing
  section. `false` keeps both copies, if you want that as a deliberate
  reinforcement strategy.

## Interaction with `protected_tags_file`

`protected_tags_file` and the attribution options are independent
mechanisms and can be combined:

- `protected_tags_file` protects specific, literal tag strings from
  `tag_dropout_percent` (e.g. character names), but has no NL-side
  equivalent and does nothing about tags_nl/nl_tags duplication.
- The attribution options above are pattern-based (not tied to one literal
  string), apply to both tags and NL, and additionally solve the
  combined-variant duplication problem.

A typical setup: put character/series tags you always want kept in
`protected_tags_file`, and rely on `attribution_dropout_immune = true` for
the "Drawn by X" style trigger specifically.

## Debugging

Set `debug_caption_processing = true` (and optionally
`debug_caption_interval`) to print the tags/NL/variant/final-caption
breakdown for sampled steps, so you can confirm the attribution ends up
where you expect before committing to a full run.