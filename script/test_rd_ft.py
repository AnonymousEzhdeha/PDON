#!/usr/bin/env python
"""Held-out test evaluation for the rd_ft experiment.

Loads each saved best_model.pth and scores it on freshly drawn test sets - the step
the training script never performs. 
"""
import runner

if __name__ == "__main__":
    runner.main_for("rd_ft")
