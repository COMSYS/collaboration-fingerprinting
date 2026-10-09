"""A drop-in replacement for the OpenAI client that logs every call.

MDAgents constructs its client as `OpenAI(api_key=...)` inside `Agent.__init__`
and then calls `self.client.chat.completions.create(...)`. We substitute the
`OpenAI` symbol in the upstream module's namespace, so no upstream source is
modified: every client the framework builds is one of ours.

Two rewrites happen on the way out, and both exist to reproduce the exact
request the published traces were collected with:

  * `model` is forced to the configured target. Upstream hardcodes model names
    in several places ('gpt-3.5' in determine_difficulty and the recruiter,
    'gpt-4o-mini' inside Group). Forcing the field here makes every call use
    one model without threading a parameter through upstream code.
  * `temperature` is dropped, because the instrumented run that produced the
    published traces sent no temperature field for this backend.

The resulting request dict is identical to the one the original inline
instrumentation sent, so request_bytes is comparable across both.
"""
import os
import time

from openai import OpenAI as _RealOpenAI

from .call_log import log_api_call


class Config:
    """Backend settings, read once from the environment.

    No endpoint is hardcoded. Our runs went to a self-hosted OpenAI-compatible
    gateway; which one is not a property of the method, and naming it in
    published code would disclose internal infrastructure for no benefit. Set
    MDA_BASE_URL to whichever endpoint serves MDA_MODEL.
    """
    model = os.environ.get('MDA_MODEL', 'gpt-oss-120b')
    base_url = os.environ.get('MDA_BASE_URL')
    api_key_env = os.environ.get('MDA_API_KEY_ENV', 'LLM_API_KEY')
    strip_temperature = os.environ.get('MDA_STRIP_TEMPERATURE', '1') == '1'


class _LoggedCompletions:
    def __init__(self, inner, owner):
        self._inner = inner
        self._owner = owner

    def create(self, **kwargs):
        kwargs['model'] = Config.model
        if Config.strip_temperature:
            kwargs.pop('temperature', None)

        role = self._owner.role
        call_start_ts = time.time_ns()
        try:
            raw = self._inner.with_raw_response.create(**kwargs)
        except Exception:
            call_end_ts = time.time_ns()
            log_api_call(role=role, call_start_ts=call_start_ts,
                         call_end_ts=call_end_ts, request_payload=kwargs,
                         response_obj=None)
            raise
        call_end_ts = time.time_ns()

        response = raw.parse()
        log_api_call(role=role, call_start_ts=call_start_ts,
                     call_end_ts=call_end_ts, request_payload=kwargs,
                     response_obj=response,
                     response_bytes=len(raw.content),
                     request_bytes=len(raw.http_response.request.content))
        return response


class _Chat:
    def __init__(self, inner, owner):
        self.completions = _LoggedCompletions(inner.chat.completions, owner)


class LoggedClient:
    """Quacks like `OpenAI` for the one call path MDAgents uses."""

    def __init__(self, *_args, **_kwargs):
        # Upstream passes api_key=os.environ['openai_api_key']; ignore it and
        # use the configured backend instead.
        if not Config.base_url:
            raise RuntimeError(
                "MDA_BASE_URL is not set. Point it at an OpenAI-compatible "
                "endpoint that serves MDA_MODEL (default 'gpt-oss-120b'), and "
                "put the key in the variable named by MDA_API_KEY_ENV "
                f"(currently '{Config.api_key_env}')."
            )
        try:
            api_key = os.environ[Config.api_key_env]
        except KeyError:
            raise RuntimeError(
                f"${Config.api_key_env} is not set. It is the variable "
                "MDA_API_KEY_ENV names as holding the API key."
            ) from None
        self._inner = _RealOpenAI(api_key=api_key, base_url=Config.base_url)
        self.role = 'unknown'
        self.chat = _Chat(self._inner, self)

    def bind_role(self, role):
        self.role = role
        return self
