#!/usr/bin/env bash
set -e

echo "=== 1) Actualizando repositorios ==="
sudo apt update

echo "=== 2) Instalando dependencias básicas ==="
sudo apt install -y wget gpg apt-transport-https software-properties-common

echo "=== 3) Añadiendo clave GPG de Microsoft para VS Code ==="
wget -qO- https://packages.microsoft.com/keys/microsoft.asc \
  | gpg --dearmor \
  | sudo tee /etc/apt/trusted.gpg.d/microsoft.gpg > /dev/null

echo "=== 4) Añadiendo repositorio oficial de VS Code ==="
sudo add-apt-repository -y "deb [arch=amd64] https://packages.microsoft.com/repos/code stable main"

echo "=== 5) Actualizando repositorios de nuevo ==="
sudo apt update

echo "=== 6) Instalando Visual Studio Code ==="
sudo apt install -y code

echo "=== 7) Instalando Python, pip y venv ==="
sudo apt install -y python3 python3-pip python3-venv

echo "=== 8) Instalando extensiones de Python para VS Code ==="
# Estas líneas pueden fallar si no hay entorno gráfico o si 'code' no está en PATH,
# por eso van con '|| true' para que el script no reviente.
code --install-extension ms-python.python || true
code --install-extension ms-toolsai.jupyter || true

echo "=== Instalación completada ==="
echo "Puedes abrir VS Code con: code"
