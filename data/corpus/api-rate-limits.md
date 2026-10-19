# API Rate Limits

The Nimbus Cloud REST API enforces rate limits per API key to protect platform stability.

## Default limits

Free tier keys are limited to 60 requests per minute. Pro tier keys are limited to 600 requests per minute. Enterprise keys default to 3,000 requests per minute and can be raised on request.

## Rate limit headers

Every response includes the headers X-RateLimit-Limit, X-RateLimit-Remaining and X-RateLimit-Reset. When a key exceeds its limit, the API returns HTTP 429 Too Many Requests with a Retry-After header in seconds.

## Best practices

Clients should retry 429 and 503 responses with exponential backoff and jitter, starting at one second and capping at 32 seconds. Batch endpoints count as a single request regardless of the number of items, up to 100 items per batch.
