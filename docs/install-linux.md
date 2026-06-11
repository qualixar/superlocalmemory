# Installing SLM on Linux (Ubuntu 22.04 / Debian)
> SuperLocalMemory V3.6.9+ | https://superlocalmemory.com

SLM requires Python **3.11 or later**. Ubuntu 22.04 ships Python 3.10 by default — use the deadsnakes PPA or a version manager.

---

## Option A — Virtual environment (recommended)

```bash
# 1. Add deadsnakes PPA and install Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-distutils -y

# 2. Create a venv
python3.11 -m venv ~/.slm-venv
source ~/.slm-venv/bin/activate

# 3. Install SLM
pip install superlocalmemory

# 4. Tell SLM which Python to use (add to ~/.bashrc or ~/.zshrc)
export SLM_PYTHON=~/.slm-venv/bin/python

# 5. Install the CLI globally
npm install -g superlocalmemory

# 6. Run setup
slm setup
```

The venv keeps SLM isolated from the system Python. `slm doctor` will verify the wiring.

---

## Option B — pipx (zero venv management)

```bash
sudo apt install pipx
pipx install superlocalmemory --python python3.11
npm install -g superlocalmemory
slm setup
```

---

## Option C — pyenv

```bash
curl https://pyenv.run | bash   # or: brew install pyenv on Linuxbrew
pyenv install 3.11.9
pyenv global 3.11.9
pip install superlocalmemory
npm install -g superlocalmemory
slm setup
```

---

## Systemd service (optional)

To start the daemon automatically at boot:

```ini
# /etc/systemd/system/slm-http.service
[Unit]
Description=SuperLocalMemory daemon
After=network.target

[Service]
Type=simple
User=<your-user>
Environment=SLM_DAEMON_HOST=127.0.0.1
Environment=SLM_DAEMON_PORT=8765
ExecStart=/home/<your-user>/.slm-venv/bin/python -m superlocalmemory.server.unified_daemon --start
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now slm-http
```

For LAN access set `SLM_DAEMON_HOST=0.0.0.0` and see the [distributed deployment guide](distributed-deployment.md).

---

## Verify

```bash
slm doctor
slm remember "test memory"
slm recall "test"
```

`slm doctor` prints a green checkmark for each gate. If Python version fails, re-check `SLM_PYTHON` and that you activated the venv.
