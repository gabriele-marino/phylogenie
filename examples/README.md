# 📚 Examples 

This folder contains a collection of well-documented configuration files for Phylogenie, organized as a step-by-step tutorial to help you become a proficient Phylogenie user.
These examples demonstrate different settings and use cases to help you get started or adapt them to your own workflows.

## 📂 Available configuration files

| File  | Read if: |
|-------|---------------|
| `1.BD-Trees.yaml` | You’re new to Phylogenie! |
| `2.BDEI-Trees.yaml` | • You want to simulate trees under a birth–death with exposed–infectious (BDEI) model;<br>• You want to sample 1-dimensional vectors;<br>• You want to run computations using your context variables. |
| `3.BDSS-Trees.yaml` | • You want to simulate trees under a birth–death with superspreading (BDSS) model;<br> • You want to create dependencies between your context variables. |
| `4.Canonical-Trees.yaml` | • You want to simulate trees under a generic multi-type birth–death (MTBD) model with a canonical parameterization;<br>• You want to learn about SkylineVectors and SkylineMatrices in Phylogenie;<br>• You want to sample multi-dimensional vectors. |
| `5.Epidemiological-Trees.yaml` | • You want to simulate trees under a generic multi-type birth–death (MTBD) model with the epidemiological parameterization. |
| `6.FBD-Trees.yaml` | • You want to simulate trees under a generic multi-type birth–death (MTBD) model with the fossilized birth-death (FBD) parameterization. |
| `7.BD-CT-Trees.yaml` | You want to add contact tracing (CT) to your tree simulations. |
| `8.BD-MSAs.yaml` | You want to simulate multiple sequence alignments (MSAs). |

## ▶️ How to use

To run Phylogenie for a given configuration file, use:

```bash
phylogenie config_file.yaml
```
Alternatively, you can run all the examples in this folder at once:

```bash
phylogenie .
```
This command will automatically detect all .yaml files in the folder and execute Phylogenie for each one.
