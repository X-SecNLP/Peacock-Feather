# Oobleck, Gödel, Hairy Ball

> **A Semantic Evolutionary Engine**:
> *Exploring the non-Newtonian fluid of language, the incompleteness of lexicons, and the inevitable singularities of vector fields.*

## The Philosophy

* **Oobleck:** Language is non-Newtonian. It resists when you strike it with rigid definitions but flows when you let it wander. This engine treats the latent space as a fluid to be navigated.
* **Gödel:** Based on the Incompleteness Theorems, no vocabulary is truly "complete." We search for the best fit within the specific "axioms" of your provided text source.
* **Hairy Ball:** In a continuous vector field, there is always a point where the "hair" stands up—a singularity. This script evolves words to find those semantic peaks where meaning converges.

## The Mechanics

This tool is a **Genetic Algorithm (GA)** explorer that "surfs" the latent space of Large Language Models. It doesn't just find synonyms; it evolves a population of thoughts.

1.  **Selection:** Words are selected based on their `util.cos_sim` (Cosine Similarity) to your query.
2.  **Niche Sharing:** To prevent the population from collapsing onto a single word too quickly, a sharing function penalizes clusters. This forces the "species" to diversify and find multiple semantic niches.

3.  **Mutation:** Random genomic drift is introduced to prevent the simulation from getting stuck in local optima.
4.  **Artifact Generation:** The convergence is rendered into a polar-coordinate animation.

## Quick Start

### 1. Requirements
```bash
pip install torch numpy matplotlib sentence-transformers

```

### 2. Run the Evolution

You can pass a list of strings or extract a unique vocabulary from any `.txt` file (lyrics, philosophy, or logs).

```python
from explorer import SemanticNicheExplorer

explorer = SemanticNicheExplorer()

# Load your custom "world"
explorer.fit_text_file('path/to/your/text.txt')

# Run the evolutionary simulation
# query: the target concept or word
history = explorer.run_evolution(query="dich", generations=100)

# Generate the .gif artifact
explorer.generate_artifact(history)

```

## Visual Interpretation

The generated `semantic_evolution.gif` uses a polar projection to map the "Thought Space":

* **Radius ($$r$$):** Represents **Fitness**. The closer a point is to the center, the more semantically "correct" it is relative to your query.
* **Angle ($$\theta$$):** Represents the **Lexical Index**. Different angles represent different regions of your vocabulary.
* **Visual Cues:** The size and glow of the particles indicate similarity, while the white vectors highlight the current "Alpha" words of the generation.
