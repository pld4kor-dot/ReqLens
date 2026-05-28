# User Stories — University Event Management System

## Epic: Event Discovery & Registration

US-001: As a student, I want to search for events by category and date so that I can
find events that interest me.
  Acceptance Criteria:
  - Search supports filtering by category, date range, and location
  - Results display within 1 second
  - Results show event title, date, location, available spots

US-002: As a student, I want to register for an event with one click so that I can
quickly sign up.
  Acceptance Criteria:
  - Registration button visible on event detail page
  - Confirmation email sent within 5 minutes
  - Registration count updates in real time

US-003: As a student, I want to be added to a waitlist when an event is full so that
I still have a chance to attend.
  Acceptance Criteria:
  - Waitlist position displayed to user
  - Automatic promotion when spot opens
  - Notification sent upon promotion

US-004: As a student, I want to receive push notifications for event updates so that
I stay informed.
  Acceptance Criteria:
  - Notifications for: event changes, cancellations, reminders
  - User can configure notification preferences
  - Notifications sent at least 24 hours before event

## Epic: Event Management

US-010: As an event organizer, I want to create an event with all required details
so that attendees have complete information.
  Acceptance Criteria:
  - Required fields: title, description, date/time, venue, capacity, category
  - Optional fields: recurring pattern, materials, co-organizers
  - Event saved as draft until explicitly published

US-011: As an event organizer, I want to set up recurring events so that I don't
have to create each occurrence manually.
  Acceptance Criteria:
  - Support weekly, bi-weekly, monthly patterns
  - Ability to edit individual occurrences
  - Ability to cancel entire series

US-012: As an event organizer, I want to upload materials (slides, handouts) so that
attendees can prepare or review.
  Acceptance Criteria:
  - Supported formats: PDF, PPTX, DOCX, images
  - Maximum file size: 50 MB
  - Materials accessible only to registered attendees

## Epic: Attendance & Academic Integration

US-020: As a faculty member, I want to link events to courses so that attendance can
count toward grades.
  Acceptance Criteria:
  - Course linking available during event creation
  - Attendance auto-recorded via QR check-in
  - Report exportable to CSV

US-021: As a faculty member, I want attendance reports by course and semester so that
I can track student participation.
  Acceptance Criteria:
  - Filterable by course, semester, date range
  - Shows per-student attendance percentage
  - Exportable to CSV and PDF

## Epic: Administration

US-030: As an administrator, I want to manage user roles so that access is properly
controlled.
  Acceptance Criteria:
  - Can assign/revoke roles: Student, Faculty, Organizer, Admin
  - Role changes take effect immediately
  - All role changes logged for audit

US-031: As an administrator, I want to view audit logs so that I can investigate
security incidents.
  Acceptance Criteria:
  - Logs include: user, action, timestamp, IP address
  - Searchable by user, action type, date range
  - Logs retained for at least 2 years

## Epic: Calendar & Social Integration

US-040: As a user, I want to sync events to my personal calendar so that I don't
miss them.
  Acceptance Criteria:
  - Export to iCal format
  - Direct sync with Google Calendar and Outlook
  - Synced events update when source event changes

US-041: As a student, I want to share events on social media so that my friends know
about them.
  Acceptance Criteria:
  - Share buttons for major platforms (Facebook, Twitter/X, WhatsApp)
  - Shared link shows event preview (Open Graph tags)
  - Shared link directs to event detail page
