#!/bin/bash
# Quick start script for gridenv_es_test_ideal_obs_repeat_task

# Exit conda environment (if active)
if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Deactivating conda environment: $CONDA_DEFAULT_ENV"
    conda deactivate 2>/dev/null || true
fi

# Check Python and JAX installation
echo "Checking Python environment..."
python3 -c "import jax; import flax; import optax; print('✓ JAX version:', jax.__version__); print('✓ Available devices:', len(jax.devices()), 'GPU(s)')" || {
    echo "Error: Required packages not installed. Please check requirements.txt"
    exit 1
}

# Run the training script
echo "Starting training..."
python3 train_evolution_strategy.py "$@"
