# Change Request CR-2025-003

## Title: Add Virtual/Hybrid Event Support

## Requester: Alex (Product Owner)
## Date: 2025-01-20
## Priority: High

## Description
Due to increasing demand for remote participation, the UEMS must support virtual
and hybrid events. This change affects event creation, registration, attendance
tracking, and the underlying infrastructure.

## Requested Changes

### 1. Event Type Extension
- Add event type field: In-Person, Virtual, Hybrid
- Virtual events require a video conferencing link (Zoom, Teams, or custom URL)
- Hybrid events have both a physical venue and a virtual link

### 2. Virtual Attendance Tracking
- For virtual events, attendance tracked via join/leave timestamps from conferencing API
- Minimum attendance duration configurable per event (e.g., must attend 75% of event)
- QR check-in not applicable for virtual attendance

### 3. Capacity Management Changes
- Virtual events: capacity limit optional (or very high, e.g., 10,000)
- Hybrid events: separate capacity for in-person and virtual attendance
- Waitlist behavior: when in-person is full, offer virtual attendance option

### 4. Infrastructure Impact
- Video conferencing API integration (Zoom API, MS Teams Graph API)
- Increased bandwidth requirements for streaming
- CDN needed for recorded session playback

### 5. User Interface Changes
- Event creation form: new fields for event type and virtual link
- Event detail page: show join link (visible only 15 min before event starts)
- Registration page: for hybrid events, let user choose in-person or virtual

## Impact Assessment (Preliminary)
- Affected requirements: FR-010, FR-020, FR-021, FR-022, FR-040, FR-041, NFR-001
- Estimated effort: 3-4 sprints
- Risk: Integration with third-party conferencing APIs may introduce reliability
  dependencies

## Approval
- [ ] IT Security review
- [ ] Architecture review
- [ ] Product Owner approval
