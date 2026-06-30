# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.1.x   | ✅ Yes    |
| < 1.1   | ❌ No     |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Email **kumaar324@gmail.com** with:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

You can expect an acknowledgement within **72 hours** and a resolution timeline within **14 days** for confirmed issues.

## Scope

This is a research/medical-adjacent tool. Areas of particular concern:
- File path traversal in CLI arguments
- Unsafe deserialization of camera pose JSON
- Arbitrary code execution via video or image inputs

## Out of Scope

- Vulnerabilities in third-party dependencies (report to the upstream project)
- Issues only reproducible on unsupported Python versions (< 3.9)
