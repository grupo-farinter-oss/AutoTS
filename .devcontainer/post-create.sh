#!/usr/bin/env bash
set -euo pipefail

# El extra "mcp" cubre las dependencias opcionales ligeras (holidays, scipy, lightgbm, etc.).
# tensorflow y prophet quedan fuera a propósito por su peso; instalarlos a mano si se necesitan.
# uv gestiona el entorno (.venv) y el lock (uv.lock); evita instalar en el site-packages del sistema.
python -m pip install --user --upgrade uv
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra mcp

# Remoto "upstream" = proyecto original, para traer cambios nuevos al fork.
if ! git remote | grep -qx upstream; then
	git remote add upstream https://github.com/winedarksea/AutoTS.git
fi

# Los repos montados desde el host suelen tener otro propietario.
git config --global --add safe.directory "$(pwd)"

uv run python -c "import autots; print('AutoTS', autots.__version__)"
