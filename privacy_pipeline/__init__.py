"""privacy_pipeline — Differential Privacy on Graph Laplacians."""

__version__ = "0.2.0"

from .config import (  # noqa: F401
    DataConfig, EmbeddingConfig, EvaluationConfig, ExperimentConfig,
    GraphConfig, NoiseConfig, OutputConfig, load_config,
)
from .evaluate import (  # noqa: F401
    AttackResult, ClassifierResult, ClassMetrics,
    display_classification_results, run_attack_all_classes,
    run_classification, run_cv_inference_attack,
)
from .experiment import (  # noqa: F401
    BaselineRecord, EpsilonRecord, ExperimentResults,
    FiedlerRecord, PrivacyExperiment,
)
from .features import (  # noqa: F401
    Dataset, extract_features, feature_columns, load_dataset,
    preprocess_features, segment_signal,
)
from .graph import (  # noqa: F401
    DPLaplacianEigenmaps, EmbeddingResult, GraphDiagnostics,
    diagnose_knn_graph, eigenvalue_perturbation,
    eigenvector_perturbation, projector_embedding_lowrank,
)
from .noise import (  # noqa: F401
    BaseNoiseMechanism, EmbeddingPerturbation, EmbeddingPerturbationMetadata,
    NoiseMetadata, PPSPLaplacianNoise, PPSPMetadata,
    ResolventGuidedPerturbation, ResolventGuidedMetadata,
    SpectralGapNoise, SpectralGapMetadata, build_noise_mechanism,
)
from .visualize import (  # noqa: F401
    plot_fiedler_evolution, plot_privacy_utility_tradeoff,
    visualize_embedding_shift, visualize_graph_perturbation,
    visualize_knn_graph,
)
from .pipelines import (  # noqa: F401
    EmbeddingNoiseMetadata, EmbeddingSpaceNoisePipeline,
    FeatureNoiseMetadata, FeatureSpaceNoisePipeline,
)
from .comparison import (  # noqa: F401
    ComparisonExperiment, ComparisonRecord, ComparisonResults,
    PipelineMetrics, plot_comparison_tradeoff, plot_pipeline_comparison,
)
