"""Training pipeline and modular loss components."""

from ham.training.pipeline import HAMPipeline, TrainingPhase
from ham.training.losses import (
    LossComponent,
    ReconstructionLoss,
    KLDivergenceLoss,
    ZermeloAlignmentLoss,
    GeodesicSprayLoss,
    VelocityDirectionAlignmentLoss,
    ContrastiveAlignmentLoss,
    MetricAnchorLoss,
    MetricSmoothnessLoss,
    LongTrajectoryAlignmentLoss,
    EulerLagrangeResidualLoss,
    AVBDPathEnergyLoss,
    WindThermodynamicLoss,
    KinematicPriorLoss,
    FinslerActionMatchingLoss,
    WindAssistedTrajectoryAlignmentLoss,
    FinslerianFlowMatchingLoss,
    ArrivalTimeLoss,
    DenseArrivalTimeLoss,
    curriculum_alpha,
)
from ham.training.losses_ebm import (
    ContrastiveDivergenceLoss,
    DenoisingScoreMatchingLoss,
    MSELoss,
)

__all__ = [
    "HAMPipeline", "TrainingPhase",
    "LossComponent",
    "ReconstructionLoss", "KLDivergenceLoss",
    "ZermeloAlignmentLoss", "GeodesicSprayLoss",
    "VelocityDirectionAlignmentLoss", "ContrastiveAlignmentLoss",
    "MetricAnchorLoss", "MetricSmoothnessLoss",
    "LongTrajectoryAlignmentLoss", "EulerLagrangeResidualLoss",
    "AVBDPathEnergyLoss", "WindThermodynamicLoss",
    "KinematicPriorLoss", "FinslerActionMatchingLoss",
    "WindAssistedTrajectoryAlignmentLoss", "FinslerianFlowMatchingLoss",
    "ArrivalTimeLoss", "DenseArrivalTimeLoss", "curriculum_alpha",
    "ContrastiveDivergenceLoss", "DenoisingScoreMatchingLoss", "MSELoss",
]
