# Security policy

## Supported versions

Security fixes go into the latest release only. Before 0.1, that is the `main` branch.

## Reporting a vulnerability

Please do not report security problems in a public issue.

Report them privately through GitHub: open the [Security tab](https://github.com/growmatics/phytotron/security) of this repository and choose **Report a vulnerability**. You need a GitHub account. GitHub's guide: [privately reporting a security vulnerability](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability).

Please include what you found, how to reproduce it, and the version or commit you used.

## What to expect

- We acknowledge your report within 5 working days.
- Within 90 days we release a fix or publish a statement on why no fix is needed.
- We credit you in the advisory and the changelog, unless you prefer not to be named.
- If a vulnerability is being actively exploited, we report it as the EU Cyber Resilience Act requires.

## Scope

In scope: the code in this repository.

Out of scope: problems in dependencies such as GreenLight-Gym2, CasADi, IPOPT or NumPy. Please report those to their own projects. If such a problem affects Phytotron, tell us as well, so we can pin or patch.

Phytotron is simulation and benchmarking software. It runs locally, offers no network service, and does not control greenhouse equipment.

## Dependencies

CI checks every dependency against known vulnerabilities (`pip-audit`). Each release includes a software bill of materials (SBOM) in CycloneDX format.

GreenLight-Gym2 is installed from its `v0.2.1` tag (commit `8bb5b28`). That tag still declares version 0.2.0 in its package metadata, so the SBOM lists `gl_gym 0.2.0`. `pip-audit` cannot check it, because it is not installed from PyPI.
