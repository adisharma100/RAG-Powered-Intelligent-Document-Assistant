# Nimbus Labs Security Policy

## Encryption
All data is encrypted in transit using TLS 1.3 and at rest using AES-256. Encryption keys are rotated every 90 days
and stored in a hardware security module.

## Access Control
Multi-factor authentication is mandatory for every employee account. Production database access is granted only to
the on-call engineering team and expires automatically after 8 hours. Passwords must be at least 14 characters long.

## Incident Response
Security incidents must be reported to security@nimbuslabs.example within 1 hour of discovery. The security team will
acknowledge the report within 30 minutes. Customers affected by a data breach are notified within 72 hours.

## Backups
Backups run every 6 hours and are retained for 35 days. Restore tests are carried out quarterly.
