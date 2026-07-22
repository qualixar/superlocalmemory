# Security Policy

## SuperLocalMemory V3 Security

### Supported Versions

| Version | Supported |
|:--------|:---------:|
| 3.7.x | Yes |
| 3.6.x | Security fixes only |
| < 3.6 | No |

### Reporting Vulnerabilities

**Do NOT open public issues for security vulnerabilities.**

Email: admin@superlocalmemory.com

Include:
- Description of the vulnerability
- Steps to reproduce
- Impact assessment
- Suggested fix (if any)

We will respond within 48 hours and provide a fix timeline within 7 days.

### Security Architecture

#### Mode A (Zero-LLM, Local-Only)
- All data stored at `~/.superlocalmemory/`
- Zero cloud API calls during store/recall operations
- No telemetry, analytics, or phone-home code
- SQLite with WAL mode for data integrity

#### Authentication
- Optional API key authentication for dashboard/API access
- HMAC-SHA256 timing-safe comparison
- Rate limiting: 30 writes/min, 120 reads/min

#### Data Protection
- Parameterized SQL queries throughout (no SQL injection)
- XSS protection via `escapeHtml()` in all UI rendering
- CSRF protection via token-based auth (no cookies)
- Security headers: X-Frame-Options, CSP, X-Content-Type-Options
- CORS whitelist with credential control

#### Model Supply Chain (Untrusted Checkpoints)

SuperLocalMemory loads local embedding, reranker, and compression models via
Hugging Face. Two risks apply:

- **`trust_remote_code`**: since v3.7.9, `trust_remote_code=True` is passed only
  for an internal allowlist of pinned models (the nomic-embed family, which
  requires custom modeling code). Any other model — including one swapped into
  config through a write path — loads with `trust_remote_code=False` and cannot
  execute repository code at load time.
- **CVE-2025-14926** (`transformers`, SEW tokenizer): arbitrary code execution
  when loading a crafted SEW tokenizer config. No upstream patch exists as of
  2026-07-20. SLM never loads checkpoints automatically; it will be upgraded as
  soon as a patch ships.

Only install models from sources you trust. A malicious checkpoint can execute
code under your user account at load time regardless of these mitigations.

#### Compliance
- GDPR Article 15 (right to access): full data export
- GDPR Article 17 (right to erasure): complete erasure including learning data
- EU AI Act data sovereignty: Mode A keeps all data local
- Tamper-proof audit trail with SHA-256 hash chain

### Dependencies

Run `npm audit` and `pip audit` regularly. Report any findings.

### Research Foundation

The security architecture is formally documented across three peer-reviewed papers:

- **Paper 1** ([arXiv:2603.02240](https://arxiv.org/abs/2603.02240)): Bayesian trust defense, OWASP-aligned memory poisoning protection
- **Paper 2** ([arXiv:2603.14588](https://arxiv.org/abs/2603.14588)): Information-geometric foundations, cellular sheaf cohomology for contradiction detection
- **Paper 3** ([arXiv:2604.04514](https://arxiv.org/abs/2604.04514)): Trust-weighted forgetting, compliance audit trails, FRQAD mixed-precision integrity

---

Part of [Qualixar](https://qualixar.com) | Author: [Varun Pratap Bhardwaj](https://varunpratap.com)
