import os
import redis
import sys
import logging
from urllib.parse import urlparse
from typing import Any
from redis.cluster import RedisCluster
from redis.client import Redis
import redis.exceptions
import redis.sentinel


def _redis_cluster_client(redis_url: str, **kwargs: Any) -> "Redis":

    return RedisCluster.from_url(redis_url, **kwargs)  # type: ignore


def _check_for_cluster(redis_client: "Redis") -> bool:
    try:
        cluster_info = redis_client.info("cluster")
        return cluster_info["cluster_enabled"] == 1
    except redis.exceptions.RedisError:
        return False


def _redis_sentinel_client(redis_url: str, **kwargs: Any) -> "Redis":
    parsed_url = urlparse(redis_url)
    # sentinel needs list with (host, port) tuple, use default port if none available
    sentinel_list = [(parsed_url.hostname or "localhost", parsed_url.port or 26379)]
    if parsed_url.path:
        # "/mymaster/0" first part is service name, optional second part is db number
        path_parts = parsed_url.path.split("/")
        service_name = path_parts[1] or "mymaster"
        if len(path_parts) > 2:
            kwargs["db"] = path_parts[2]
    else:
        service_name = "mymaster"

    sentinel_args = {}
    if parsed_url.password:
        sentinel_args["password"] = parsed_url.password
        kwargs["password"] = parsed_url.password
    if parsed_url.username:
        sentinel_args["username"] = parsed_url.username
        kwargs["username"] = parsed_url.username

    # check for all SSL related properties and copy them into sentinel_kwargs too,
    # add client_name also
    for arg in kwargs:
        if arg.startswith("ssl") or arg == "client_name":
            sentinel_args[arg] = kwargs[arg]

    # sentinel user/pass is part of sentinel_kwargs, user/pass for redis server
    # connection as direct parameter in kwargs
    sentinel_client = redis.sentinel.Sentinel(
        sentinel_list, sentinel_kwargs=sentinel_args, **kwargs
    )

    # redis server might have password but not sentinel - fetch this error and try
    # again without pass, everything else cannot be handled here -> user needed
    try:
        sentinel_client.execute_command("ping")
    except redis.exceptions.AuthenticationError:
        exception_info = sys.exc_info()
        exception = exception_info[1] or None
        if exception is not None and "no password is set" in exception.args[0]:
            logging.warning(
                msg="Redis sentinel connection configured with password but Sentinel \
    answered NO PASSWORD NEEDED - Please check Sentinel configuration"
            )
            sentinel_client = redis.sentinel.Sentinel(sentinel_list, **kwargs)
        else:
            raise

    return sentinel_client.master_for(service_name)


def _get_client(redis_url: str, pool=False, **kwargs: Any) -> "Redis":
    # Initialize with necessary components.
    redis_client: "Redis"
    # check if normal redis:// or redis+sentinel:// url
    if redis_url.startswith("redis+sentinel"):
        redis_client = _redis_sentinel_client(redis_url, **kwargs)
    elif redis_url.startswith("rediss+sentinel"):  # sentinel with TLS support enables
        kwargs["ssl"] = True
        if "ssl_cert_reqs" not in kwargs:
            kwargs["ssl_cert_reqs"] = "none"
        redis_client = _redis_sentinel_client(redis_url, **kwargs)
    else:
        # connect to redis server from url, reconnect with cluster client if needed
        redis_client = redis.from_url(redis_url, **kwargs)
        if _check_for_cluster(redis_client):
            redis_client.close()
            redis_client = _redis_cluster_client(redis_url, **kwargs)
    return redis_client


def _create_redis_pool():
    return redis.ConnectionPool.from_url(
        os.getenv("G_REDIS_URL", "redis://localhost:6379/0")
    )


pool = _create_redis_pool()


def get_redis():
    return redis.StrictRedis(connection_pool=pool)
