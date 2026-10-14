"""Environment registry (GEM is optional for standalone inference).

Agentic scaffolding derives from https://github.com/RAGEN-AI/RAGEN.
"""
try:
    import gem
except ModuleNotFoundError as exc:
    if exc.name != "gem":
        raise
else:
    gem.register("city_tool", entry_point="roll.pipeline.agentic.env.city_grounding:CityGroundingEnv")
