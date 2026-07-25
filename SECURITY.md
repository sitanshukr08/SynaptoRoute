# Security Policy

## Scope & Security Boundary

SynaptoRoute is an edge-native, local semantic pre-routing dispatch library. It is designed to route incoming text queries to workflow intents, tool handlers, or downstream LLM components based on semantic similarity.

**What SynaptoRoute Is:**
- A local pre-routing optimization layer.
- An in-memory vector index and local SQLite persistence engine.

**What SynaptoRoute Is NOT:**
- A security authorization or access control boundary.
- An input sanitization or prompt injection barrier.
- A firewall or authentication gateway.

Applications using SynaptoRoute must perform security authorization, prompt validation, and parameter sanitization downstream within the target tool or workflow handler.

## Reporting Vulnerabilities

If you discover a potential security issue in SynaptoRoute, please report it responsibly by contacting the maintainers directly or opening a confidential security advisory.

Please include:
1. Description of the vulnerability and impact.
2. Step-by-step reproduction code or script.
3. Affected versions and environment metadata.
