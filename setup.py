#!/usr/bin/env python3
"""
Project bootstrap script for Part 2: Multi-Agent Simulated World.

This script creates the recommended folder structure and placeholder files
in the project root. It is intended to be run ONCE when initializing the repo.

Usage:
    python setup.py

Note:
- This is NOT a packaging setup.py for pip.
- It is a project scaffolding / initializer script.
"""

from pathlib import Path

# -----------------------------
# Project structure definition
# -----------------------------

PROJECT_STRUCTURE = {
    "configs": [
        "world.yaml",
        "agents.yaml",
        "training.yaml",
    ],
    "src": {
        "environment": [
            "__init__.py",
            "world.py",
            "resources.py",
            "clans.py",
            "dynamics.py",
            "gym_wrapper.py",
        ],
        "agents": [
            "__init__.py",
            "base_agent.py",
            "emotion.py",
            "risk.py",
            "observation.py",
        ],
        "training": [
            "__init__.py",
            "train_ppo.py",
            "callbacks.py",
            "evaluate.py",
        ],
        "gui": [
            "__init__.py",
            "app.py",
            "renderer.py",
            "colors.py",
        ],
        "utils": [
            "__init__.py",
            "logger.py",
            "seeding.py",
            "metrics.py",
        ],
        "__init__.py": None,
        "main.py": None,
    },
    "models": {
        "trained": [],
        "checkpoints": [],
    },
    "results": {
        "plots": [],
        "logs": [],
        "videos": [],
    },
    "notebooks": [
        "analysis.ipynb",
        "ablations.ipynb",
    ],
    "scripts": [
        "train.sh",
        "evaluate.sh",
        "demo.sh",
    ],
}

ROOT_FILES = [
    "README.md",
    "requirements.txt",
    ".gitignore",
]

# -----------------------------
# Helper functions
# -----------------------------


def create_file(path: Path, content: str = ""):
    if not path.exists():
        path.write_text(content)
        print(f"[+] Created file: {path}")


def create_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"[+] Created directory: {path}")


# -----------------------------
# Recursive structure creation
# -----------------------------


def create_structure(base: Path, structure):
    if isinstance(structure, dict):
        for name, sub in structure.items():
            path = base / name
            if sub is None:
                create_file(path)
            else:
                create_dir(path)
                create_structure(path, sub)
    elif isinstance(structure, list):
        for item in structure:
            create_file(base / item)


# -----------------------------
# Main execution
# -----------------------------


def main():
    root = Path.cwd()

    print("\nInitializing Part 2 project structure...\n")

    # Root-level files
    for file in ROOT_FILES:
        create_file(root / file)

    # Main directories
    create_structure(root, PROJECT_STRUCTURE)

    print("\nProject structure successfully initialized.\n")
    print("Next steps:")
    print("  1. Create and activate a virtual environment (venv)")
    print("  2. Install dependencies and update requirements.txt")
    print("  3. Start implementing the environment in src/environment/")


if __name__ == "__main__":
    main()
