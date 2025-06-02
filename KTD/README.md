# CTD Test Runner

This utility allows you to run **x86-based container threat detection test cases** in Google Kubernetes Engine to validate Security Command Center Container Threat Detection (CTD) alerts.

---

## 🧾 Prerequisites

- Python 3.6+
- `gcloud` CLI installed and authenticated
- `kubectl` configured for your GKE cluster
- Premium/Enterprise tier of SCC for full logging visibility

---

## 🗂 Files

- `ctd_test_runner.py` — Main CLI tool
- `ctd_tests.json` — Maintains test definitions (easy to extend)

---

## 🔐 Authentication

### Option 1: Use a JWT
```bash
python3 ctd_test_runner.py --jwt /path/to/creds.json --all
