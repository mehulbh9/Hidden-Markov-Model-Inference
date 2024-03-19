# Hidden Markov Model Inference

This repository contains the `inference.py` script, which implements algorithms for performing inference in hidden Markov models (HMMs). The script includes functions for computing forward and backward messages, calculating marginals, and finding the most likely sequence of states using the Viterbi algorithm.

## Project Description

The `inference.py` script is designed to work with HMMs where states are not directly observable. Instead, observations provide some information about the states. The script can handle cases where some observations may be missing.

## Features

- Computes forward messages using the forward algorithm.
- Computes backward messages using the backward algorithm.
- Calculates marginal distributions at each time step.
- Implements the Viterbi algorithm to find the most likely sequence of hidden states.
