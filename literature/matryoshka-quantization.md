# Matryoshka Quantization

URL: https://arxiv.org/abs/2502.06786

## Citation Metadata

- Authors: Pranav Nair, Puranjay Datta, Jeff Dean, Prateek Jain, Aditya Kusupati.
- Year: 2025.
- Status: arXiv preprint.

## Core Idea

Matryoshka Quantization uses the nested structure of integer bit representations so lower bitwidth models can be extracted from the most significant bits of higher bitwidth representations.

## Relevance

This is an important substrate for adaptive precision. If lower and higher precisions are nested, runtime precision selection can be implemented more naturally than switching among unrelated quantized checkpoints.

## What It Suggests For This Project

- Consider bit-slicing or residual-bit routing as an implementation path.
- Compare uniform precision selection with adaptive selection on the same nested representation.
- Account for the fact that lower precision slices may not behave like independently optimized low-bit quantizers.

## Open Questions For Follow-Up

- Does nested quantization make dynamic routing easier enough to offset any accuracy loss?
- Are token- or layer-level bit slices executable efficiently in existing kernels?
