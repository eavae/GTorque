import os
import redis


def _create_redis_pool():
    return redis.ConnectionPool(
        host=os.getenv("G_REDIS_HOST", "localhost"),
        port=os.getenv("G_REDIS_PORT", 6379),
        db=os.getenv("G_REDIS_DB", 0),
    )


pool = _create_redis_pool()


def get_redis():
    return redis.StrictRedis(connection_pool=pool)
