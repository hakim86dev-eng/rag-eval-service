# Data Retention and Backups

Managed Postgres databases are backed up automatically every 24 hours, and point-in-time recovery is available for the last 7 days on the Pro tier and the last 35 days on the Enterprise tier.

## Data residency

Customers choose a region when creating a project. Data never leaves the selected region, except for encrypted backups, which are replicated to a paired region within the same jurisdiction. The Middle East region is hosted in Dubai and paired with Riyadh.

## Deletion

When a project is deleted, its data is soft-deleted and can be restored by support for 7 days. After 7 days, data is permanently erased, and backups containing it expire within 35 days.

## Logs

Application logs are retained for 30 days by default. Audit logs are retained for 400 days and can be exported to customer-owned storage.
