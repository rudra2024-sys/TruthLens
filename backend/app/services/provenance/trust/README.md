# C2PA trust anchors

`c2pa_trust_list.pem` is the official C2PA trust list (the certificate authorities whose signatures count as
trusted Content Credentials issuers), copied unmodified except that the human-readable `Subject` header lines
were dropped and only the PEM certificate blocks kept.

- Source: https://raw.githubusercontent.com/c2pa-org/conformance-public/main/trust-list/C2PA-TRUST-LIST.pem
- Retrieved: 2026-09-26 (30 certificates)
- Refresh it periodically: the list changes as issuers are added or revoked. A stale list can only make the
  app report `untrusted` for a newer issuer - it never makes an untrusted signer look trusted.
