docker run --restart unless-stopped -p 6379:6379 -d  --name redis redis:latest
celery -A main worker -l info
