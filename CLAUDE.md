---
description: Project overview and architecture for TreE-Graph
---

# TreE-Graph Project Overview

TreE-Graph simplifies scikit-learn decision trees using egglog e-graphs.

## Core Approach

1. **Extract** - Convert trained sklearn trees to egglog terms (`If(condition, then, else)`, `Leaf(value)`)
2. **Saturate** - Apply rewrite rules to discover equivalent tree structures
3. **Extract** - Extract the minimal equivalent tree from the e-graph

## Architecture

- **Core module**: `src/` - E-graph conversion and simplification logic
- **Tests**: `tests/` - Unit and integration tests
- **Examples**: `examples/` - Usage demonstrations

## Documentation

See @README.md for extensive project overview
