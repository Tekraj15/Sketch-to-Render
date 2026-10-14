"""Evaluation harness for Sketch-to-Render (WS1).

Two-phase design: `generate` renders a benchmark with one engine and records
timings, `score` loads scorers one at a time and writes metrics + a report.
Nothing here modifies the engines; seeding and timing are applied from outside.
"""
