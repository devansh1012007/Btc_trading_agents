"""
Expert Layer (§6): runs every registered expert for one bar and assembles
their outputs into a single ExpertOutputs.
"""
from typing import Dict

from ..experts.base_expert import BaseExpert
from ..state.contracts import ExpertOutputs, FeatureVector, RegimeState, ZoneMap


class ExpertLayer:
    def __init__(self, experts: Dict[str, BaseExpert]):
        self.experts = experts

    def evaluate(self, fv: FeatureVector, zones: ZoneMap, regime: RegimeState) -> ExpertOutputs:
        outputs = ExpertOutputs(timestamp=fv.timestamp, symbol=fv.symbol)
        for expert in self.experts.values():
            outputs = outputs.add_expert(expert.evaluate(fv, zones, regime))
        return outputs
