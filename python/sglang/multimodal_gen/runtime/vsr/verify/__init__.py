# SPDX-License-Identifier: Apache-2.0
"""VSR migration acceptance tooling.

These are one-off comparison tools for the VSR migration, not runtime code:
they exist to prove that the SGLang implementation matches the reference
implementation bit-for-bit where it must and within tolerance where it may
drift. See ``docs_always/add_new_mode/add_vsr/requirements.md`` §4.

The reference implementation lives in an external, read-only repository, so
everything here observes it through monkey-patching rather than editing it.
"""
