# University IT Security & Data Privacy Policy (Excerpt)

## Section 5: Application Security Requirements

5.1 Authentication
All university applications MUST use the centralized SSO service backed by the
institutional LDAP directory. Local authentication mechanisms are prohibited for
production systems.

5.2 Authorization
Applications MUST implement role-based access control (RBAC). Role definitions
must be approved by the IT Security Office before deployment.

5.3 Data Encryption
- Data at rest: AES-256 encryption required for all personally identifiable
  information (PII) stored in databases or file systems.
- Data in transit: TLS 1.2 or higher required for all network communications.
  TLS 1.3 is recommended.

5.4 Session Management
- Session tokens must be invalidated after 30 minutes of inactivity.
- Concurrent sessions per user must be limited to 5.
- Session tokens must use secure, HttpOnly cookies.

5.5 Audit Logging
All administrative and security-relevant actions must be logged with:
- Timestamp (UTC)
- User identity
- Action performed
- Source IP address
- Result (success/failure)

Audit logs must be retained for a minimum of 2 years and stored in a tamper-resistant
format.

## Section 7: Data Privacy (GDPR Compliance)

7.1 Data Minimization
Applications shall collect only the minimum personal data necessary for the stated
purpose. Data collection must be justified in the Data Protection Impact Assessment.

7.2 User Rights
Applications must support:
- Right to access (Article 15): Users can request a copy of their personal data.
- Right to rectification (Article 16): Users can correct inaccurate data.
- Right to erasure (Article 17): Users can request deletion of their data.
- Right to portability (Article 20): Data export in machine-readable format.

7.3 Consent
Where processing is based on consent, the system must:
- Obtain explicit opt-in consent (no pre-checked boxes).
- Allow withdrawal of consent at any time.
- Record the timestamp and scope of each consent.

7.4 Data Retention
Personal data must be deleted or anonymized when no longer needed for its original
purpose, and no later than 3 years after last user activity.

## Section 9: Availability & Business Continuity

9.1 Availability Targets
- Tier 1 (Critical): 99.9% uptime
- Tier 2 (Important): 99.5% uptime
- Tier 3 (Standard): 99.0% uptime

9.2 Backup & Recovery
- RPO: Must not exceed 1 hour for Tier 1 and Tier 2 systems.
- RTO: Must not exceed 4 hours for Tier 1, 8 hours for Tier 2.
- Backups must be stored in a geographically separate location.
