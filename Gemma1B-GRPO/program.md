## Goal
Your goal is to find the best finetuning configuration through experimentation to achieve the highest "Post-RL total reward" (which you can find in the last line of the output).

## How you work
You start in a clean and dedicated branch (e.g., `fg-finetune/YYYY-MM-DD-hash`, you can make changes only in this branch.) and begin with a baseline run by invoking `python run.py`.  Then you propose changes to `run.py` and run various experiments to achieve the best result. It is essentially an infinite loop like this:

```
LOOP FOREVER:

1. Look at the current git state: the current branch/commit we're on
2. Tune `run.py` with an experimenal idea by directly hacking the code
3. git commit
4. Run the experiment: `python run`
5. Record the result in `results.tsv` (see below). But do not commit `result.tsv`
6. If the "Final Post-train Eval Accuracy" improves after finetuning, you "advance" the branch, keeping the git commit
7. If the "Final Post-train Eval Accuracy" is equal or worse, you git reset to where you started
```

DO NOT stop until you completely run out of ideas and can not come up with anything new to try.

Each experimental run should take 2-3 hours (do not try to shorten it; let it be). Sometimes it crashes due to various reasons. Try some quick fixes if it does, and move on if you cannot fix it.

DO NOT look at previous Git commit messages of the 'main' branch because it's not your concern.

## Logging results
When an experiment is done, log the result to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:
```
commit Post_RL_total_accuracy numerical_accuracy format_accuracy status(keep/discard) description
```
Note: commit is 7 characters


## What you CAN change:
The `run.py` file is the only file you can touch. Feel free to experiment with:

*LoRA parameters (rank, alpha)
*Target LoRA layers
*Random seed (try multiple)
*Optimizer (try AdamW, Muon, SGD etc.)
*Learning rate (try 1e-6, 5e-6, 1e-5 etc.)
*Learning rate schedule (try 'constant', 'cosine', 'linear')
*Batch size (larger or smaller)
*Context length (longer or shorter)
*Initialization methods
* Sampling paramaeters for training (temperature, top_k, top_p)
*Other hyperparameters as you see fit

## What you CANNOT change:
*MODEL_ID
*dataset(GSM8K)
*TRAIN_FRACTION
*Any reward or score function
*NUM_EPOCHS
*NUM_ITERATIONS
*NUM_GENERATIONS
*NUM_BATCHES
*Generation config ('greedy')
*Checkpoint config
*Gemma model architecture
*Any other file
