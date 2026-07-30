"""
ML Adversarial Attacks Framework for SUMO VANETs
Implements 6 selected attacks: 4 critical + 2 defense tests
"""

from .universal_perturbation import UniversalPerturbationAttack
from .sumo_adapter import UniversalPerturbationSUMOAdapter, create_adapter
# from .hopskipjump import HopSkipJumpAttack
# from .clean_label_backdoor import CleanLabelBackdoorAttack
# from .badnets import BadNetsAttack
# from .membership_inference import MembershipInferenceAttack
# from .copycat_cnn import CopycatCNNAttack

__all__ = [
    "UniversalPerturbationAttack",
    "UniversalPerturbationSUMOAdapter",
    "create_adapter",
    # "HopSkipJumpAttack",
    # "CleanLabelBackdoorAttack",
    # "BadNetsAttack",
    # "MembershipInferenceAttack",
    # "CopycatCNNAttack",
]
