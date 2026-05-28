"""Unit test fixtures and configuration."""

import pytest


@pytest.fixture
def sample_transcript_text() -> str:
    """A small stakeholder interview transcript for testing."""
    return """\
Alice: We need a system where students can register for university events.

Bob: Yes, and they should be able to log in with their Google account.

Alice: The system should send email notifications when an event is updated.

Bob: We also need an admin panel where staff can create and manage events.

Alice: What about availability? I think the system should be available 24/7.

Bob: Security is important too. We need to make sure student data is protected.

Alice: Events should support at most 500 participants.

Bob: And the system should load pages quickly, ideally under 2 seconds.

Alice: We should allow students to cancel registrations up to 24 hours before the event.

Bob: One more thing – we need to comply with GDPR for data handling.
"""


@pytest.fixture
def sample_srs_text() -> str:
    """A small legacy SRS for testing."""
    return """\
# University Event Management System – SRS

## 1. Introduction
This document describes the requirements for the University Event Management System (UEMS).

## 2. Functional Requirements

### FR-001: Event Registration
The system shall allow students to register for events.

### FR-002: Google Login
The system shall support login via Google OAuth 2.0.

### FR-003: Email Notifications
The system shall send email notifications when event details change.

### FR-004: Admin Panel
Staff members shall be able to create, edit, and delete events through an admin interface.

### FR-005: Registration Cancellation
Students shall be able to cancel event registrations up to 24 hours before the event start time.

## 3. Non-Functional Requirements

### NFR-001: Availability
The system shall maintain 99.9% uptime.

### NFR-002: Performance
All pages shall load within 2 seconds under normal load (up to 1000 concurrent users).

### NFR-003: Security
Student personal data shall be encrypted at rest and in transit.

### NFR-004: Capacity
Each event shall support a maximum of 500 registered participants.

### NFR-005: Compliance
The system shall comply with GDPR requirements for personal data handling.
"""


@pytest.fixture
def sample_change_request_text() -> str:
    """A sample change request."""
    return "Replace Google login with institutional SSO (Shibboleth/SAML)."


@pytest.fixture
def hallucination_candidate_text() -> str:
    """A candidate requirement NOT supported by the demo sources."""
    return "The system shall be available 24/7 with zero downtime."
