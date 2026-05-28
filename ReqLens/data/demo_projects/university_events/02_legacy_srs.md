# University Event Management System — Legacy SRS (v0.9 Draft)

## 1. Introduction

### 1.1 Purpose
This document specifies the software requirements for the University Event
Management System (UEMS), a web-based platform for managing campus events.

### 1.2 Scope
UEMS enables students, faculty, and administrators to create, discover, register
for, and manage events across the university campus.

## 2. Functional Requirements

### 2.1 User Management
- FR-001: The system shall authenticate users via university SSO (LDAP).
- FR-002: The system shall support four roles: Student, Faculty, Organizer, Admin.
- FR-003: The system shall allow administrators to assign and revoke roles.

### 2.2 Event Management
- FR-010: The system shall allow authorized users to create events with title,
  description, date/time, venue, capacity, and category.
- FR-011: The system shall support event categories: Academic, Social, Sports,
  Cultural, Career, and Other.
- FR-012: The system shall support recurring events with configurable patterns.
- FR-013: The system shall allow event organizers to edit or cancel events.
- FR-014: The system shall send notifications to all registrants upon event
  cancellation.

### 2.3 Registration
- FR-020: The system shall allow users to register for events.
- FR-021: The system shall enforce event capacity limits.
- FR-022: The system shall maintain a waitlist when capacity is reached.
- FR-023: The system shall automatically promote waitlisted users when spots open.
- FR-024: The system shall send email confirmation upon registration.
- FR-025: The system shall allow users to cancel their registration.

### 2.4 Event Discovery
- FR-030: The system shall provide search by keyword, category, date range, and
  location.
- FR-031: The system shall display a calendar view of upcoming events.
- FR-032: The system shall recommend events based on user preferences and history.

### 2.5 Attendance Tracking
- FR-040: The system shall support QR-code-based check-in.
- FR-041: The system shall generate attendance reports by event, course, and semester.
- FR-042: Faculty shall be able to link events to specific courses.

### 2.6 Content Management
- FR-050: Organizers shall be able to upload event materials (PDF, PPTX, images).
- FR-051: Materials shall be accessible only to registered attendees.
- FR-052: The system shall support a maximum file size of 50 MB per upload.

### 2.7 Calendar Integration
- FR-060: The system shall export events to iCal format.
- FR-061: The system shall support sync with Google Calendar and Outlook.

### 2.8 Social Features
- FR-070: Users shall be able to share events on social media platforms.
- FR-071: Users shall see which of their contacts are attending an event.

## 3. Non-Functional Requirements

### 3.1 Performance
- NFR-001: The system shall support 5000 concurrent users.
- NFR-002: Page response times shall not exceed 2 seconds under normal load.
- NFR-003: Event search shall return results within 1 second.

### 3.2 Availability
- NFR-010: The system shall maintain 99.5% uptime during academic semesters.
- NFR-011: Planned maintenance windows shall be scheduled outside peak hours.

### 3.3 Security
- NFR-020: All data shall be encrypted at rest (AES-256) and in transit (TLS 1.3).
- NFR-021: The system shall log all administrative actions for audit purposes.
- NFR-022: The system shall comply with GDPR requirements.
- NFR-023: Session tokens shall expire after 30 minutes of inactivity.

### 3.4 Usability
- NFR-030: The interface shall be responsive and mobile-friendly.
- NFR-031: The system shall support English, German, and French.
- NFR-032: The system shall conform to WCAG 2.1 AA accessibility standards.

### 3.5 Reliability
- NFR-040: Recovery Point Objective (RPO) shall be 1 hour.
- NFR-041: Recovery Time Objective (RTO) shall be 4 hours.

## 4. Constraints
- C-001: The system shall be built using modern web technologies.
- C-002: The system shall integrate with the existing university IT infrastructure.
- C-003: The MVP shall be delivered by the start of the spring semester.
