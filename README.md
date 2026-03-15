# NOMAD: A Multi-Agent LLM System for UML Class Diagram Generation from Natural Language Requirements

This repository contains code and data from the paper NOMAD: A Multi-Agent LLM System for UML Class Diagram Generation from Natural Language Requirements.

> **NOMAD: A Multi-Agent LLM System for UML Class Diagram Generation from Natural Language Requirements**  
> [[Paper]](https://arxiv.org/abs/2511.22409) [[PDF]](https://arxiv.org/pdf/2511.22409)  
> *Authors: Polydoros Giannouris, Sophia Ananiadou*

## Evaluation

Use the evaluation utilities to compare a generated PlantUML class diagram against a reference (golden) diagram.

### Usage

```python
from src.eval_helpers import evaluate_uml

results = evaluate_uml(golden_uml, generated_uml)
print(results)
