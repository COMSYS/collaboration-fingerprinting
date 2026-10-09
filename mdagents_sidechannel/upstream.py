"""Load the upstream MDAgents `utils` module, fixed and instrumented.

MDAgents is third-party code with no license granting redistribution, so this
repository contains none of it. Instead we read the module from a local clone,
apply a small number of targeted corrections to its source in memory, and
execute the result. Nothing is written back to the clone.

Three corrections are applied. Each one fixes a crash that stops a run partway
through; all three were needed to collect the published traces, so they are
required to reproduce them. The anchors below are the minimum text needed to
locate each site, and loading fails loudly if upstream no longer matches --
silently skipping a fix would change the data a run produces.

A fourth upstream defect needs no correction here: `utils.py` contains a
backslash inside an f-string expression, which is a SyntaxError before Python
3.12 and legal from 3.12 on (PEP 701). We require >= 3.12 instead of editing it.
"""
import importlib.util
import os
import sys
import types
from pathlib import Path

from .instrumentation import Config, LoggedClient

MIN_PYTHON = (3, 12)

# (description, anchor, replacement)
FIXES = [
    (
        "recruiter output that does not match 'N. role - description' raises "
        "IndexError and kills the run",
        r"""    for i, agent in enumerate(agents_data):
        agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
        description = agent[0].split('-')[1].strip().lower()
        agent_list += f"Agent {i+1}: {agent_role} - {description}\n"
""",
        r"""    for i, agent in enumerate(agents_data):
        try:
            agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
            description = agent[0].split('-')[1].strip().lower()
        except IndexError:
            continue
        agent_list += f"Agent {i+1}: {agent_role} - {description}\n"
""",
    ),
    (
        "an expert may name an agent number that does not exist, indexing "
        "medical_agents out of range",
        r"""                    chosen_experts = [int(ce) for ce in chosen_expert.replace('.', ',').split(',') if ce.strip().isdigit()]
""",
        r"""                    chosen_experts = [int(ce) for ce in chosen_expert.replace('.', ',').split(',') if ce.strip().isdigit()]
                    chosen_experts = [ce for ce in chosen_experts if 1 <= ce <= len(medical_agents)]
""",
    ),
    (
        "a recruited MDT with no parseable members produces an empty Group "
        "that later code cannot use",
        r"""        group_instance = Group(res_gs['group_goal'], res_gs['members'], question)
""",
        r"""        if not res_gs['members']:
            print(f"[WARN] Group {i1+1} had no parseable members in the recruiter's response; skipping.")
            continue

        group_instance = Group(res_gs['group_goal'], res_gs['members'], question)
""",
    ),
]


def mdagents_home():
    env = os.environ.get('MDAGENTS_HOME')
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / 'third_party' / 'MDAgents')


def apply_fixes(src):
    """Return `src` with every fix applied, or raise if a site is missing."""
    applied = []
    for desc, anchor, replacement in FIXES:
        count = src.count(anchor)
        if count != 1:
            raise RuntimeError(
                f"Cannot apply fix ({desc}): expected exactly 1 match of its "
                f"anchor in utils.py, found {count}. The upstream checkout is "
                f"not the pinned commit, or has been modified. Refusing to "
                f"continue, because running without this fix changes the data."
            )
        src = src.replace(anchor, replacement, 1)
        applied.append(desc)
    return src, applied


def load(verbose=True):
    """Import upstream `utils` with fixes and instrumentation applied."""
    if sys.version_info < MIN_PYTHON:
        raise RuntimeError(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required: upstream "
            f"utils.py has a backslash inside an f-string expression, which "
            f"only parses from 3.12 on (PEP 701). Found "
            f"{sys.version_info.major}.{sys.version_info.minor}."
        )

    home = mdagents_home()
    utils_path = home / 'utils.py'
    if not utils_path.is_file():
        raise RuntimeError(
            f"No MDAgents checkout at {home}. Run ./setup_mdagents.sh, or set "
            f"MDAGENTS_HOME to an existing clone."
        )

    src, applied = apply_fixes(utils_path.read_text(encoding='utf-8'))

    mod = types.ModuleType('utils')
    mod.__file__ = str(utils_path)
    spec = importlib.util.spec_from_loader('utils', loader=None)
    mod.__spec__ = spec
    sys.modules['utils'] = mod

    old_cwd = os.getcwd()
    sys.path.insert(0, str(home))
    try:
        exec(compile(src, str(utils_path), 'exec'), mod.__dict__)
    finally:
        sys.path.remove(str(home))
        os.chdir(old_cwd)

    _instrument(mod)

    if verbose:
        print(f"[upstream] loaded {utils_path}")
        for desc in applied:
            print(f"[upstream]   fix applied: {desc}")
        print(f"[upstream] all calls routed to {Config.model} at {Config.base_url}")
    return mod


def _instrument(mod):
    """Replace the OpenAI symbol and bind each Agent's role to its client."""
    mod.OpenAI = LoggedClient

    # Upstream reads os.environ['openai_api_key'] as an *argument* to OpenAI(),
    # so that lookup happens before our replacement is called and would raise
    # KeyError. The value is discarded by LoggedClient, which authenticates
    # with Config.api_key_env instead; this placeholder only satisfies the read.
    os.environ.setdefault('openai_api_key', 'unused-by-instrumented-client')

    orig_init = mod.Agent.__init__

    def __init__(self, *args, **kwargs):
        model_info = kwargs.get('model_info')
        if model_info is None and len(args) >= 4:
            model_info = args[3]
        if model_info == 'gemini-pro':
            raise NotImplementedError(
                "gemini-pro is not wired into the per-call logging schema "
                "(request/response byte accounting is not implemented for the "
                "google.generativeai SDK, which does not go through our client). "
                "Refusing to make an unlogged network call rather than leave a "
                "silent gap in the trace."
            )
        orig_init(self, *args, **kwargs)
        client = getattr(self, 'client', None)
        if isinstance(client, LoggedClient):
            client.bind_role(self.role)

    mod.Agent.__init__ = __init__
