# Test Cases — University Event Management System

## TC-001: User Authentication via SSO
Precondition: User has valid university LDAP credentials
Steps:
  1. Navigate to login page
  2. Click "Login with University SSO"
  3. Enter LDAP credentials
  4. Submit
Expected: User is authenticated and redirected to dashboard with correct role

## TC-002: Event Creation (Happy Path)
Precondition: User is logged in as Organizer
Steps:
  1. Navigate to "Create Event"
  2. Fill in: Title="Guest Lecture: AI Ethics", Date="2025-03-15 14:00",
     Location="Auditorium A", Capacity=200, Category="Academic"
  3. Add description
  4. Click "Publish"
Expected: Event is created, visible in search, organizer receives confirmation

## TC-003: Event Registration with Capacity Limit
Precondition: Event exists with capacity=2, 1 spot remaining
Steps:
  1. User A registers → succeeds (capacity reached)
  2. User B registers → added to waitlist
  3. User A cancels registration
Expected: User B is automatically promoted from waitlist and notified via email

## TC-004: Event Search by Category and Date
Precondition: Multiple events exist across categories
Steps:
  1. Navigate to search page
  2. Filter by category="Academic" and date range="March 2025"
  3. Submit search
Expected: Only academic events in March 2025 are displayed, within 1 second

## TC-005: Recurring Event Creation
Precondition: User is logged in as Organizer
Steps:
  1. Create event with recurring pattern "Weekly on Monday"
  2. Set end date 4 weeks from start
Expected: 4 event occurrences are created, each editable independently

## TC-006: QR Code Check-In
Precondition: User is registered for event, event is happening now
Steps:
  1. Organizer generates QR code for event
  2. Attendee scans QR code with mobile device
  3. System records check-in
Expected: Attendance recorded with timestamp, reflected in attendance report

## TC-007: Event Cancellation Notification
Precondition: Event has 50 registered attendees
Steps:
  1. Organizer cancels event
  2. System processes cancellation
Expected: All 50 registrants receive cancellation email within 15 minutes

## TC-008: File Upload for Event Materials
Precondition: Event exists, user is event organizer
Steps:
  1. Navigate to event management page
  2. Upload a 25 MB PDF file
  3. Verify upload success
  4. Access as registered attendee → should succeed
  5. Access as non-registered user → should fail
Expected: File uploaded, access controlled by registration status

## TC-009: Calendar Sync (iCal Export)
Precondition: User is registered for event
Steps:
  1. Click "Add to Calendar" on event detail page
  2. Select iCal format
  3. Import downloaded .ics file into calendar app
Expected: Event appears in personal calendar with correct details

## TC-010: GDPR Data Export
Precondition: User is logged in, has event history
Steps:
  1. Navigate to Privacy settings
  2. Click "Export My Data"
  3. Download generated archive
Expected: Archive contains all personal data in machine-readable format (JSON)

## TC-011: Performance Under Load
Type: Non-functional / Load Test
Setup: Simulate 5000 concurrent users
Steps:
  1. 3000 users browsing events
  2. 1500 users performing searches
  3. 500 users registering for events simultaneously
Expected: All page response times under 2 seconds, no errors

## TC-012: Session Timeout
Precondition: User is logged in
Steps:
  1. Leave session idle for 30 minutes
  2. Attempt to navigate to a protected page
Expected: User is redirected to login page with session expired message

## TC-013: Role-Based Access Control
Precondition: Users with different roles exist
Steps:
  1. Student tries to access admin panel → denied
  2. Faculty creates academic event → allowed
  3. Admin assigns organizer role → allowed
  4. Organizer tries to manage other's event → denied
Expected: Each action enforced according to role permissions

## TC-014: Attendance Report Generation
Precondition: Course-linked events have attendance data
Steps:
  1. Faculty navigates to Reports
  2. Selects course "CS101" and semester "Spring 2025"
  3. Generates attendance report
Expected: Report shows per-student attendance %, exportable to CSV
