# Examples

This folder contains a collection of well-documented configuration files for Phylogenie, organized as a step-by-step tutorial to help you become a proficient Phylogenie user.
These examples demonstrate different settings and use cases to help you get started or adapt them to your own workflows.


## Available Configuration Files

- `1-TreeSimulatorBD.yaml` \
   Learn the basics of Phylogenie, and use is it to generate a dataset of trees with a lognormally distributed reproduction number and infectious period.
- 


## How to Use

To run Phylogenie for a given configuration file, use:

```bash
phylogenie <config_file>.yaml
```
Alternatively, you can run all the examples in this folder at once:

```bash
phylogenie .
```
This command will automatically detect all .yaml files in the folder and execute Phylogenie for each one.
