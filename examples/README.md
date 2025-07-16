# 📚 Examples 

This folder contains a collection of well-documented configuration files for Phylogenie, organized as a step-by-step tutorial to help you become a proficient Phylogenie user.
These examples demonstrate different settings and use cases to help you get started or adapt them to your own workflows.

## 📂 Available configuration files

| File  | Read if: |
|-------|---------------|
| `1-TreeSimulatorBD.yaml` | You’re new to Phylogenie! |
| `2-TreeSimulatorBDEI.yaml` | • You want to simulate trees under a birth–death with exposed–infectious (BDEI) model;<br>• You want to sample 1-dimensional vectors;<br>• You want to run computations using your context variables. |
| `3-TreeSimulatorMTBD.yaml` | • You want to simulate trees under a generic multi-type birth–death (MTBD) model;<br>• You want to learn about SkylineVectors and SkylineMatrices in Phylogenie;<br>• You want to sample multi-dimensional vectors;<br>• You want to run computations using your context variables. |
| `4-AliSimBD.yaml` | You want to simulate multiple sequence alignments (MSAs) using the AliSim backend. |


## ▶️ How to use

To run Phylogenie for a given configuration file, use:

```bash
phylogenie <config_file>.yaml
```
Alternatively, you can run all the examples in this folder at once:

```bash
phylogenie .
```
This command will automatically detect all .yaml files in the folder and execute Phylogenie for each one.
