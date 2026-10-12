# antar-demo-target

A minimal ML training repo used to demo the **antar** agent (neatHack).

It intentionally has a gap: AdamW + cosine schedule + mixed precision, but
**no gradient clipping** — antar should detect it, verify it against the
literature, open an issue, and propose a PR.
