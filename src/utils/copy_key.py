#!/usr/bin/env python3

import os
import sys
import re
from pathlib import Path

# Read full input from stdin
input_data = sys.stdin.read()

# Extract RSA private key block
match = re.search(
    r"-----BEGIN RSA PRIVATE KEY-----.*?-----END RSA PRIVATE KEY-----",
    input_data,
    re.DOTALL
)

if not match:
    print("No valid RSA private key found in input.", file=sys.stderr)
    sys.exit(1)

private_key = match.group(0)

# Define path for the key
ssh_dir = Path.home() / ".ssh"
key_path = ssh_dir / "computa_key"

# Ensure ~/.ssh exists
ssh_dir.mkdir(mode=0o700, exist_ok=True)

# Write key to file
with open(key_path, "w") as f:
    f.write(private_key)

# Set strict permissions
os.chmod(key_path, 0o600)

print(f"Private key written to {key_path}")
