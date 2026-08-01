# Security policy

Moonshine is a research preview that loads very large local model files,
allocates most of a qualified host's memory, reads checkpoint data through
direct I/O, and exposes an optional HTTP service. Do not expose it to an
untrusted network without an API key and additional network controls.

## Supported versions

| Version | Support |
|---|---|
| `main` | Best effort |
| `0.2.x` research preview | Security fixes planned |
| `0.1.x` research preview | Superseded by `0.2.x` |

## Reporting a vulnerability

Use GitHub's private vulnerability reporting feature:

1. Open the repository's **Security** tab.
2. Choose **Advisories** and **Report a vulnerability**.
3. Include reproduction steps, impact, affected revision, and any suggested
   mitigation.

Do not include vulnerability details in a public issue. If private reporting
has not yet been enabled, open a public issue containing no sensitive details
and ask the repository owner for a private contact method.

Maintainers will acknowledge a report as soon as practical, coordinate a fix
and disclosure timeline, and credit reporters who want attribution.

Model-weight licensing, model behavior, prompt injection, and the security of
the upstream checkpoint are outside this code repository's security boundary.
